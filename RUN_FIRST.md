# 直接用 sbatch 提交独立任务

## 你现在只需要提交训练

作业 59210 失败的原因：旧目录 `outputs/pano_official_full/` 已存在 `training_config.json`，训练器拒绝覆盖。该目录没有 checkpoint，不能用 `--resume` 恢复。旧文件已保留，没有删除或覆盖。

数据索引和文本特征已经完成，所以不用重新准备数据。你当前在 `DIT360/` 目录，执行：

```bash
mkdir -p qwen_pano/outputs
sbatch qwen_pano/scripts/train_pano.sh
```

不需要先激活环境，脚本自己激活 `qwen360`。新作业会使用独立目录 `qwen_pano/outputs/pano_official_full_<作业号>/`，不会再使用旧失败目录。

当前 train_pano.sh 将日志写到相对于提交目录的 `qwen_pano/outputs/qwen-train_<作业号>.log`。请统一从 `DIT360/` 提交，避免产生嵌套的 qwen_pano 目录。Slurm 在运行脚本前打开日志，因此提交前创建目录。

```bash
tail -f qwen_pano/outputs/qwen-train_60000.log
```

## 每个脚本只执行一个任务

| 脚本 | 唯一任务 | 当前是否需要 |
|---|---|---|
| `prepare_data.sh` | 索引已有 polished Arrow 缓存，不导出图片 | 已完成，跳过 |
| `cache_text.sh` | 缓存 Qwen 文本特征，不训练 | 已完成，跳过 |
| `train_pano.sh` | 从头进行官方全量全景 LoRA 训练 | **现在提交这个** |
| `resume_pano.sh` | 从指定完整 checkpoint 继续原训练 | 有 checkpoint 后使用 |
| `zero_shot.sh` | 原始 Qwen 零样本推理 | 需要时提交 |
| `inference_pano.sh` | 加载指定 LoRA checkpoint 推理 | 训练后使用 |
| `train_mix.sh` | 官方 mix 配置训练，需要额外透视数据和文本缓存 | 当前不用 |
| `train_paper.sh` | 论文正文配置训练，需准备额外数据并声明未公开参数 | 当前不用 |

所有 `.sh` 都有自己的 `#SBATCH` 配置、Conda 激活和一条 Python 任务命令；不调用其他项目 shell 文件。旧的 submit.sh、run_stage.sh、_common.sh、finetune_cached.sh 和 qwen_job.sbatch 已移除，避免继续误用旧入口。

## 后续常用命令

以下统一从 `DIT360/` 执行：

```bash
# 新环境才需要依次准备；每一步成功后再提交下一步，不会自动排依赖。
sbatch qwen_pano/scripts/prepare_data.sh
sbatch qwen_pano/scripts/cache_text.sh

# 新训练：每次新作业使用新输出目录
sbatch qwen_pano/scripts/train_pano.sh

# 零样本推理
sbatch qwen_pano/scripts/zero_shot.sh

# 恢复训练：替换为真实存在且含 COMPLETE.json 的 checkpoint
sbatch qwen_pano/scripts/resume_pano.sh qwen_pano/outputs/pano_official_full_60000/checkpoint-epoch003

# 加载训练结果推理
sbatch qwen_pano/scripts/inference_pano.sh qwen_pano/outputs/pano_official_full_60000/checkpoint-epoch025
```

传给 Python 的相对路径统一相对于 `DIT360/`，不相对于 scripts/。恢复训练会继续原来的输出目录；不要修改总 epochs 为剩余 epochs。没有完整 checkpoint 时应重新提交 train_pano.sh，从新目录重训；新脚本不会假装恢复未保存的进度。

## 必要配置放在哪里

- 环境：每个脚本内 `conda activate qwen360`。脚本使用 `CONDA_EXE` 或 PATH 中的 conda，不写死环境安装目录。
- 项目：从提交目录向上定位含 `qwen_pano/train.py` 的目录，不写个人绝对路径。请从项目内提交。
- 数据：读取现有 Hugging Face 缓存配置。官方全量 10,359 条，不剔除样本。
- 资源：GPU 任务指定 GPU4、72 小时；训练/恢复/mix/paper 申请 256G 主机内存，其他任务保留 70G。训练为单 GPU、26 CPU，workers=25。prepare_data.sh 只索引数据，不申请 GPU。
- 模型：默认 Qwen/Qwen-Image-2512。更换模型时缓存与训练必须同步，不能混用文本特征。
- 网络：保留 HF 离线默认值。确需下载模型时，提交前设置 `export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0`。
- 输出：训练和推理使用作业号区分目录；索引、文本缓存保持原路径，可复用。

数值训练配方不变：2048×1024、lr=5e-5、25 epoch、batch=1、累积=4、workers=25、rank/alpha=64、dropout=0.05、padding=1、cube/yaw=0.5。完整 polished 与现有 benchmark 的身份重叠仍会记录，不作独立泛化测试结论。

`TRANSFORMERS_CACHE` FutureWarning 不是这次退出原因，不需要通过屏蔽警告解决。另一个旧作业 59083 在 GPU3 的 44.42 GiB 显卡上出现过 CUDA OOM；更换输出目录解决的是目录冲突，不保证显存问题也随之解决。本次保留当前 GPU4 配置，没有更改训练分辨率或精度来降低显存。

没有替你提交作业或运行模型。

## 作业 59217 的主机内存 OOM

Slurm 报告 oom_kill，并杀死 DataLoader worker。这与 CUDA 显存 OOM 不同。训练脚本的 70G 已提高到 256G，作为重新运行的初始预算，并非实测峰值或官方要求；是否足够仍需实际作业验证。当前全量全景训练已改用 Hugging Face load_dataset 读取官方 train；不再通过自定义 ArrowImages 读取训练图片。旧索引保留用于核对样本身份、顺序和文本缓存。

不要删除 --mem 来尝试自动扩容：本机 debug 分区 DefMemPerCPU=1000，26 CPU 默认通常只申请 26000 MiB。全局 UNLIMITED 不覆盖分区默认，也不等于作业无限内存。--mem=0 请求整个节点的内存，不适合当前共享节点。最终以 scontrol show job 的 AllocTRES 为准。

59217 目录目前只有配置与训练日志，没有完整 checkpoint。复用已有数据索引与文本缓存，直接提交新训练；不需要重新导出或缓存。代码修改会改变训练器记录的源码哈希，旧代码 checkpoint 的严格恢复校验不能通过，不应绕过该校验。

官方 DiT360/train.sh 没有 Slurm 资源配置。它使用 FLUX、Lightning/DeepSpeed stage 2、16-mixed；本项目使用 Qwen、Accelerate、BF16 与 FP32 VAE、预缓存文本特征和 Hugging Face load_dataset 读取，不能称为逐项一致的官方实现。官方 DataLoader persistent_workers=True 且复用 loader；本项目纯全景训练也复用同一个 loader，并在 workers>0 时启用 persistent_workers=True（workers=0 调试模式禁用）。保留的公开全景配方为 25 epoch / lr 5e-5 / 累积 4，而论文附录 C 为 20 epoch / lr 2e-5 / 累积 3 / 5 张 H20；两者本身不同。当前 train_pano.sh 对齐公开全景脚本的可迁移配方。

## 当前纯全景数据加载

train_pano.sh / resume_pano.sh 的 official-full 配置调用 `load_dataset("Insta360-Research/Matterport3D_polished", revision=索引中记录的版本, split="train", keep_in_memory=False)`，复用 Hugging Face 缓存，不导出图像。启动时核对完整数据行数、每行 caption 与索引顺序。现有 JSONL 作为样本身份与文本缓存校验记录继续保留；已有 text embedding cache 不需要重算。

纯全景 DataLoader 在 epoch 循环外创建一次，25 个 worker 持续复用，保留 pin_memory 和 drop_last。随机打乱使用独立采样生成器；worker 通过共享 epoch 状态刷新增强随机种子，避免恢复训练时依赖未保存的 worker 随机状态。随机序列与旧版/DiT360 不保证逐位一致。Accelerate、BF16 Transformer、FP32 VAE、损失和数值训练参数保持原设置。mix 入口不参与当前任务。

本次源码改变后应新建训练作业；旧源码 checkpoint 无法通过严格源码一致性校验。提交方式不变：在 DIT360 根目录运行 `mkdir -p qwen_pano/outputs`，然后 `sbatch qwen_pano/scripts/train_pano.sh`。

## LoRA 推理强度对照

`inference.py` 新增 `--lora-scale`，默认 1.0；脚本已有参数转发，无需修改启动脚本。建议固定 checkpoint、seed、28 步、CFG=4 和 padding 模式，仅改变 scale。每个强度使用单独目录，例如在 DIT360 根目录：

```bash
sbatch qwen_pano/scripts/inference_pano.sh \
  qwen_pano/outputs/pano_official_full_59263/checkpoint-epoch005 \
  --steps 28 --limit 20 --lora-scale 0.5 \
  --output qwen_pano/outputs/preview_epoch005_scale0.5_s28_n20
```

scale=0 关闭 LoRA 修正，但仍保留全景 padding，不等于原生 zero-shot。新目录的 generation_config.json 记录 lora_scale；变更强度必须使用不同目录。

本次新增 resume_compat.py 和 lora_scale_source_migration.json，仅允许已记录的完整源码哈希从修改前迁移到本次修改后；不放宽训练参数或任意源码变化的检查。旧 checkpoint 可继续恢复，旧生成配置缺少 scale 时按 1.0（无 LoRA 则为空）解释，续跑不改写其配置或已有图片元数据。此前“任意源码修改必须重训”的说明对此次精确兼容迁移不适用。这两个兼容文件应保留并随项目上传。
