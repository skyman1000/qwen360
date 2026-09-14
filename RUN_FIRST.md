# 直接用 sbatch 提交独立任务

## 你现在只需要提交训练

作业 59210 失败的原因：旧目录 `outputs/pano_official_full/` 已存在 `training_config.json`，训练器拒绝覆盖。该目录没有 checkpoint，不能用 `--resume` 恢复。旧文件已保留，没有删除或覆盖。

数据索引和文本特征已经完成，所以不用重新准备数据。你当前在 `DIT360/` 目录，执行：

```bash
cd qwen_pano/scripts
sbatch train_pano.sh
```

不需要先激活环境，脚本自己激活 `qwen360`。新作业会使用独立目录 `qwen_pano/outputs/pano_official_full_<作业号>/`，不会再使用旧失败目录。

日志直接写入提交目录，例如作业号为 60000：

```bash
tail -f qwen-train_60000.log
```

也可以从 `qwen_pano/` 执行 `sbatch scripts/train_pano.sh`；日志就在该提交目录。没有预先创建 logs 子目录的要求，因为 Slurm 在脚本开始执行前就需要打开日志。

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

以下从 `qwen_pano/scripts/` 执行：

```bash
# 新环境才需要依次准备；每一步成功后再提交下一步，不会自动排依赖。
sbatch prepare_data.sh
sbatch cache_text.sh

# 新训练：每次新作业使用新输出目录
sbatch train_pano.sh

# 零样本推理
sbatch zero_shot.sh

# 恢复训练：替换为真实存在且含 COMPLETE.json 的 checkpoint
sbatch resume_pano.sh qwen_pano/outputs/pano_official_full_60000/checkpoint-epoch003

# 加载训练结果推理
sbatch inference_pano.sh qwen_pano/outputs/pano_official_full_60000/checkpoint-epoch025
```

传给 Python 的相对路径统一相对于 `DIT360/`，不相对于 scripts/。恢复训练会继续原来的输出目录；不要修改总 epochs 为剩余 epochs。没有完整 checkpoint 时应重新提交 train_pano.sh，从新目录重训；新脚本不会假装恢复未保存的进度。

## 必要配置放在哪里

- 环境：每个脚本内 `conda activate qwen360`。脚本使用 `CONDA_EXE` 或 PATH 中的 conda，不写死环境安装目录。
- 项目：从提交目录向上定位含 `qwen_pano/train.py` 的目录，不写个人绝对路径。请从项目内提交。
- 数据：读取现有 Hugging Face 缓存配置。官方全量 10,359 条，不剔除样本。
- 资源：每个脚本顶部明确保留 GPU4、72 小时、70G；训练为单 GPU、26 CPU，workers=25。prepare_data.sh 只索引数据，不申请 GPU。
- 模型：默认 Qwen/Qwen-Image-2512。更换模型时缓存与训练必须同步，不能混用文本特征。
- 网络：保留 HF 离线默认值。确需下载模型时，提交前设置 `export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0`。
- 输出：训练和推理使用作业号区分目录；索引、文本缓存保持原路径，可复用。

数值训练配方不变：2048×1024、lr=5e-5、25 epoch、batch=1、累积=4、workers=25、rank/alpha=64、dropout=0.05、padding=1、cube/yaw=0.5。完整 polished 与现有 benchmark 的身份重叠仍会记录，不作独立泛化测试结论。

`TRANSFORMERS_CACHE` FutureWarning 不是这次退出原因，不需要通过屏蔽警告解决。另一个旧作业 59083 在 GPU3 的 44.42 GiB 显卡上出现过 CUDA OOM；更换输出目录解决的是目录冲突，不保证显存问题也随之解决。本次保留当前 GPU4 配置，没有更改训练分辨率或精度来降低显存。

本次只改脚本、检查目录与日志并做静态/模拟启动检查，没有替你提交作业或运行模型。
