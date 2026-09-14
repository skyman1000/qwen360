# Qwen-Pano：DiT360 训练方案移植与零样本 ERP 评测

本目录对应 `idea` 中的 Qwen zero-shot ERP 与后续 panorama adaptation。输出为宽×高 `2048×1024`。默认 `Qwen/Qwen-Image-2512`，可切换 `Qwen/Qwen-Image`；不支持 Edit/Lightning 等其他模型结构。

**当前直接执行：从 `qwen_pano/scripts/` 运行 `sbatch train_pano.sh`。数据索引和文本缓存已经完成，训练脚本只负责训练。每个 .sh 独立完成一个任务，不再使用多层分发入口。具体用途、日志和恢复命令见 [RUN_FIRST.md](RUN_FIRST.md)。**

`qwen360` 是 Conda 环境名，`qwen_pano/` 是项目包目录，二者无需同名。每个批处理脚本自己激活 `qwen360` 并定位项目；直接 `sbatch` 提交，无需填写个人绝对路径。

**先读 [执行说明 RUN_FIRST.md](RUN_FIRST.md)**：说明已经修改了什么、该选哪个训练脚本，以及各阶段的 `srun` / `sbatch` 命令。

需要核对来源时看 [审计报告 AUDIT.md](AUDIT.md)：上一版并不严格对应原始训练配置，本次已逐文件核对并修正。官方论文、panorama shell、mix shell 的设置并不一致，现在分开配置。Qwen 的必要接口差别、明确的原代码修错和未公开的数据/训练细节均在报告中列出，不声称获得作者未发布的完整配方。

交付仅通过静态源码与 shell 语法检查；未执行依赖安装、数据生成、单元测试、推理或训练。新增测试由你手动运行。

## 1. 代码入口与来源

| 文件 | 用途 |
|---|---|
| `inference.py` | 原始 Qwen / panorama LoRA 推理；复用现有 1092 条 prompt |
| `profiles.py` | 分别固定 repo-panorama、repo-mix、paper 与 custom 配置 |
| `train.py` | Qwen 原生 flow matching、attention LoRA、分阶段训练与 epoch 恢复 |
| `data.py` | 官方 JSONL 字段兼容、图像增强、mask、动态混合 |
| `cached_data.py` | 直接索引已有 Arrow 图像；训练按行读取，不复制图片 |
| `prepare_data.py` | polished 数据导出；透视照片投影到 ERP + mask |
| `cache_text.py` | 冻结 Qwen 文本编码器的离线缓存 |
| `losses.py`、`vendor/dit360/` | flow/yaw/cube，训练几何直接复用固定版官方源码 |
| `circular.py`、`pipeline.py` | 圆周 token/RoPE；训练临时 pad，推理持续 padded latent |
| `diagnostics.py` | seam 统计、cubemap、透视视图和人工检查表 |
| `audit_static.py`、`tests/` | 静态溯源检查及待你执行的 CPU 数值/接口测试 |

官方版本：[`3779fe7965473f6824994c663a0ae7a76bc7aafa`](https://github.com/Insta360-Research-Team/DiT360/tree/3779fe7965473f6824994c663a0ae7a76bc7aafa)。13 个关键文件与本地 DiT360 逐字节一致，哈希见 `upstream_sources.json`。Qwen 接口固定到 Diffusers 0.37.0。

## 2. 独立环境

以下命令从项目根目录 `DIT360/` 手动执行，不修改已有 DiT/评估环境。

```bash
conda activate qwen360
# 根据服务器 CUDA 环境选择 PyTorch wheel，再安装其余依赖。
python -m pip install -r qwen_pano/requirements.txt
python -m pip check
```

默认使用完整 BF16 Qwen transformer；不自动量化。repo/paper 训练的 VAE 按上游设为 FP32，LoRA 参数也为 FP32。训练优先使用大显存 GPU；LoRA 不会消除冻结主干的显存占用。推理可以 CPU offload。依赖组合尚未做实际安装验证。

建议用 `--revision <Hub commit>` 固定模型，或传 `--model /absolute/snapshots/<commit>`。已有完整缓存可加 `--local-files-only`。不同 Qwen 主干各自缓存文本和训练 LoRA，不能混用。

## 3. Qwen zero-shot：先运行这个

默认读取现有 `benchmark_assets/mp3d_stitched1092/prompts.jsonl`，**不改 prompt**，保留其中 `This is a panorama.`。样本 ID、逐样本 seed 算法和分辨率对应 `DiT360/inference_benchmark.py`。

```bash
# 小批次使用独立目录。
sbatch qwen_pano/scripts/zero_shot.sh \
  --limit 8 --output qwen_pano/outputs/zero_shot_2512_pilot

# 完整 1092 条。
sbatch qwen_pano/scripts/zero_shot.sh --output qwen_pano/outputs/zero_shot_2512
```

等价入口：

```bash
python -m qwen_pano.inference \
  --model Qwen/Qwen-Image-2512 \
  --prompts benchmark_assets/mp3d_stitched1092/prompts.jsonl \
  --output qwen_pano/outputs/zero_shot_2512 \
  --width 2048 --height 1024 --steps 50 --true-cfg-scale 4 \
  --seed 0 --seed-mode per-id --prompt-mode verbatim --offload model
```

原生 zero-shot 不安装任何全景 padding，不拉伸/补缝生成结果。Qwen 使用 `true_cfg_scale` 与空格 negative prompt；不能把这个数值等同于 FLUX 的 guidance embedding。默认 50 步 vs DiT 基线的 28 步不是等预算比较；可另用 `--steps 28` 在新目录做预算消融。

`--offload none` 适用于显存充足的 GPU；`--vae-tiling` 是需单独记录的推理设置。想比较更强 ERP 提示，单独传 `--prompt-mode erp-prefix`，新目录输出；不替换原文 prompt baseline。额外 seed 同样每组一个目录。

输出为图片、同名 JSON、`generation_config.json`；全部完成后才写 `generated.jsonl`。文件名仍为 `<id>_seed<seed>.png`，评估 `prompt` 保持原文，实际提示在 `generation_prompt`。记录 snapshot、版本、代码哈希、图片 SHA256、耗时和 peak allocated VRAM。

同命令可续跑；配置或已有图片校验不符时报错，不混入旧结果。每个输出目录只允许一个进程。强制终止遗留 `.generation.lock` 时，先确认其中 PID/Slurm job 已退出，再手动删除 lock。

## 4. 评估全景能力

```bash
python -m qwen_pano.diagnostics \
  --manifest qwen_pano/outputs/zero_shot_2512/generated.jsonl \
  --output qwen_pano/outputs/diagnostics_zero_shot --preview-count 24

python -m qwen_pano.diagnostics \
  --manifest DiT360/outputs/mp3d_stitched1092_perid_g3_s28/generated.jsonl \
  --output qwen_pano/outputs/diagnostics_dit360 --preview-count 24
```

诊断提供 seam 居中图、六面 cubemap、yaw180 预览、透视视图及人工表格。低 seam 误差也可能来自模糊/纯色图，不代表真实 360°；roll 后再逆 roll 也不能证明模型 yaw 一致性。需要检查整体房间闭合、跨 seam 家具身份、重复物体、极区、六面透视结构及文本符合度。

继续使用原来的评估 Python、权重与参数进行 FID/CS/QA 等对比：

```bash
# 在原有评估环境运行，非新建 Qwen 环境。
cd DiT360
python -m evaluation.evaluate \
  --generated-manifest ../qwen_pano/outputs/zero_shot_2512/generated.jsonl \
  --reference-manifest ../benchmark_assets/mp3d_stitched1092/reference.jsonl \
  --reference-label "Same MP3D stitched1092 reference as DiT360" \
  --output ../qwen_pano/outputs/quality_zero_shot \
  --metrics fid cs brisque niqe --clip-model openai/clip-vit-base-patch16
cd ..
```

完整 11 项配置见 `DiT360/evaluation/README.md`；固定区域裁剪、cube 面尺寸、FAED 权重和预处理。本目录不新增虚构的 KID、深度/法线一致性或因果指标。模型选择另设按房屋隔离的 validation，不反复用测试集调参。

## 5. 微调数据结构

### Panorama：作者发布的 polished 数据

```bash
python -m qwen_pano.prepare_data polished \
  --output qwen_pano/data/polished
```

使用官方 `Insta360-Research/Matterport3D_polished` 的 `train`：图像和 caption 原样导出为 PNG/JSONL，额外生成全白 mask、来源索引、provenance。无需重新使用别的模型修补极区。

```text
polished/
├── images/*.png
├── masks/full_white.png
├── train.jsonl
└── provenance.json
```

兼容原 mix 源码的每行三个字段：

```json
{"image":"images/example.png","caption":"A furnished living room.","mask":"masks/full_white.png"}
```

这是格式示例，不是真实 MP3D 样本。也可附加 `id/scene_id/scan_id/split`；没有 ID 时生成稳定记账 ID，**不把哈希当作真实房屋身份**。mixed caption 为字符串列表时，与上游一样选最长项。

公开 polished 顶层 schema 只有 image/caption，但本地 Parquet 的 `image.path` 保留全景 UUID。**2026-09-13 全量身份核查发现：官方 polished 中 1,077 个 UUID 与当前 1,092 条 benchmark 重合。** 导出器现保留 `source_image_path/source_view_id`，训练器拒绝已知 UUID、样本 ID 或房屋重叠。`--allow-unverified-split` 只允许身份未知的记录，不能绕过已知重叠。详见 [身份核查结果](local_identity_check.json)。

若使用普通训练入口评估独立泛化，应先重新划分数据。当前按用户选择使用官方全量，提交 `train_pano.sh`；它通过带 provenance 的全量索引和显式 `--official-full-training-only` 声明允许重叠，训练配置标记 `official_full_training_only_no_independent_test_claim`。`--allow-unverified-split` 本身依然不能绕过已知重叠。可选按房屋排除测试房屋的本地划分不属于作者原始全量训练集。

有可信映射时可给导出器 `--scene-map mapping.jsonl`，字段为 `row_index/id/scene_id/split`，覆盖该 revision 全部行；随后自行按房屋划分 train/val/test manifests。训练的 `--heldout-manifest` 默认是现有 1092 prompt，可换为包含全部 validation/test 房屋的 JSONL。

repo/paper 按源代码默认启用 panorama 随机水平翻转与 yaw，caption 不变；这是源代码行为。未来引入“左/右/前/后”和 world-state 条件时，改用 `custom --no-augment` 或实现条件协同变换；不要直接沿用原增强。

### Perspective：外部透视图先投影

论文的该分支为外部高质量 landscape 图像，经中心正方形裁剪、侧面 cube→ERP 重投影得到 RGB+mask，不是把普通图像拉伸成 2:1。

```bash
python -m qwen_pano.prepare_data perspective \
  --input /absolute/path/to/landscape_sources.jsonl \
  --output qwen_pano/data/projected_perspective \
  --projection-mode paper --height 1024 --fov 90 --seed 0 --invalid-fill 0
```

输入每行需要 `id/image/caption`，MP3D 派生图片还应保留 scene_id。输出 `perspective.jsonl` 与 RGB/mask；白色区域参与监督。`--invalid-fill 0` 是例子中声明的黑背景选择，作者未公布其默认值。随机 yaw 分布和像素中心插值也明确属于本地预处理实现，不能宣称与作者预生成文件逐像素相同。

默认 paper 投影固定正方形、90°侧面、1024高；其他尺寸/FOV、`--keep-aspect` 须选 `--projection-mode custom`。mask 不再额外腐蚀；训练使用 nearest resize 到 latent 尺寸和 `>128` 阈值。原源码中 PIL mask 宽高写反的问题已修复并在审计报告注明。

**官方未公开论文的完整 40k 透视照片、caption 与投影文件清单。自备数据可以用于同类适配实验，但不是同一训练集。** polished 的合成极区也不能作为未来世界状态的真实几何标签。

## 6. 缓存文本

单独运行冻结的 Qwen 文本编码器，训练进程只读取 safetensors，减少驻留显存。两个 manifest 的 caption 统一处理，模型 snapshot、文本长度和版本必须与训练一致。

```bash
# Panorama-only。
python -m qwen_pano.cache_text \
  --model Qwen/Qwen-Image-2512 \
  --manifests qwen_pano/data/polished/train.jsonl \
  --output qwen_pano/cache/text_2512

# Hybrid 使用另外的缓存目录，同时传入两类数据。
python -m qwen_pano.cache_text \
  --model Qwen/Qwen-Image-2512 \
  --manifests qwen_pano/data/polished/train.jsonl \
              qwen_pano/data/projected_perspective/perspective.jsonl \
  --output qwen_pano/cache/text_2512_mix
```

## 7. 选择有明确来源的训练配置

| profile | 适用目的 | lr / epochs / accumulation | 训练 pad / cube / yaw |
|---|---|---|---|
| `repo-panorama`（默认） | 对应发布 `train.sh` 的全景适配基线 | 5e-5 / 25 / 4 | 1 / 0.5 / 0.5 |
| `repo-mix` | 对应发布 mix shell；它不是论文完整几何配置 | 5e-5 / 25 / 5 | 0 / 0 / 0 |
| `paper` | 论文明确设置 + 对缺失细节的主动声明 | 2e-5 / 20 / 3 | 1 / 显式提供 / 显式提供 |
| `custom` | 小规模调试或消融 | 可改 | 可改 |

repo/paper 中固定项不能被冲突参数覆盖。训练器为 Accelerate 单卡/DDP，Qwen transformer BF16；原项目为 Lightning/DeepSpeed2、FLUX FP16。硬件、分布式实现和 backbone 的差别无法称为同一实验，详情见审计报告。

**对应官方 panorama 脚本：**

```bash
python -m qwen_pano.train --profile repo-panorama \
  --panorama-manifest qwen_pano/data/polished/train.jsonl \
  --text-cache qwen_pano/cache/text_2512 \
  --output qwen_pano/outputs/pano_repo
```

保留 full ERP flow objective、前2个 epoch warmup后加 yaw/cube、attention Q/K/V/out LoRA、rank/alpha64、dropout0.05、AdamW、5% step LR warmup。VAE FP32、数据增强开、drop_last=True、不额外 gradient clipping。`scripts/train_pano.sh` 是独立 sbatch 入口，使用已完成的官方全量缓存索引和 `--official-full-training-only`；上面的 Python 例子仍是普通 manifest 入口。

**对应官方 mix shell：**

```bash
python -m qwen_pano.train --profile repo-mix \
  --panorama-manifest qwen_pano/data/polished/train.jsonl \
  --perspective-manifest qwen_pano/data/projected_perspective/perspective.jsonl \
  --text-cache qwen_pano/cache/text_2512_mix \
  --output qwen_pano/outputs/pano_repo_mix
```

epoch0/1只用 panorama；epoch2起加入 `0.5**(epoch-1)` 的 perspective 子集，至少1张；perspective 有效区域均值损失权重0.5。LR按epoch从0.2倍升至1倍。几何项按源代码只在epoch<3启用，但发布shell的系数实际为0。

**论文主方法：**

```bash
# 示例：0.5、constant、full 是这里明确选定的缺失参数，不能称为已核实论文值。
python -m qwen_pano.train --profile paper \
  --panorama-manifest /absolute/path/to/verified_train.jsonl \
  --perspective-manifest qwen_pano/data/projected_perspective/perspective.jsonl \
  --text-cache /absolute/path/to/matching_text_cache \
  --output qwen_pano/outputs/pano_paper \
  --lambda-cube 0.5 --lambda-yaw 0.5 \
  --paper-lr-schedule constant --paper-mask-reduction full
```

该profile每个epoch遍历两类完整数据，从开始使用几何项；这一调度是论文未披露细节的声明性实现选择。`full`按整幅网格平均masked error，`valid`按有效区域平均；分母选择与loss系数需一起报告。源码存在已知bug，不为“严格”二字复制它们。

论文原文用5卡、每卡1、累积3，有效batch15；本地实际卡数会写入配置。如果使用2卡DDP，不能仍报告batch15。多卡启动示例：

```bash
accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
  --module qwen_pano.train --profile repo-panorama \
  --panorama-manifest qwen_pano/data/polished/train.jsonl \
  --text-cache qwen_pano/cache/text_2512 \
  --output qwen_pano/outputs/pano_repo_ddp
```

每卡完整复制主干，不支持把配置换成FSDP/DeepSpeed后直接复用本保存流程。

先做小规模运行检查或消融时，显式选custom：

```bash
python -m qwen_pano.train --profile custom \
  --panorama-manifest /absolute/path/to/verified_small_train.jsonl \
  --text-cache /absolute/path/to/matching_text_cache \
  --output qwen_pano/outputs/smoke_custom \
  --height 512 --epochs 1 --workers 0 --no-augment \
  --geometry-backend upstream --lambda-cube 0 --lambda-yaw 0 --padding-columns 0
```

custom默认关闭增强、VAE BF16、常数LR，可改rank、尺寸、epochs等，不标为原版配置。只要启用custom的yaw/flip增强，需要无方向依赖的panorama caption并标记 `orientation_invariant:true`。

## 8. 保存、恢复、适配后推理

每个完整epoch保存 `checkpoint-epochNNN`，包括 Diffusers LoRA（含alpha metadata）、PEFT恢复权重、pano/training配置、optimizer/scheduler/RNG；所有rank完成后写 `COMPLETE.json`。冻结20B主干不重复保存。

`--resume checkpoint` 从下个epoch继续，**epochs也必须与原计划相同**，因为上游LR warmup依赖总epochs。中断最多损失当前epoch。若要延长训练、改分辨率或改方案，使用新输出目录、`--profile custom --init-lora checkpoint`，这是只加载权重的新阶段。

```bash
python -m qwen_pano.inference \
  --model Qwen/Qwen-Image-2512 \
  --lora qwen_pano/outputs/pano_repo/checkpoint-epoch025 \
  --output qwen_pano/outputs/adapted_repo \
  --width 2048 --height 1024 --steps 50 --true-cfg-scale 4 \
  --seed 0 --seed-mode per-id --prompt-mode verbatim --offload model
```

新的schema2 checkpoint默认使用persistent padded推理：扩展latent参与全部scheduler steps、按扩展token数计算shift，末尾裁掉halo再交给Qwen VAE。训练时仍是临时padding/crop。原DiT推理硬编码1列，因此repo-panorama/repo-mix都默认推理pad1，**repo-mix的训练pad0与推理pad1是源代码本身的区别**。

旧schema1 checkpoint仍按其旧temporary语义加载，不静默改变结果。`--padding-mode temporary/persistent`、`--padding-columns 0/1`可在新目录显式做推理消融。训练cache的文本长度也会随checkpoint读取。base snapshot必须匹配。

## 9. 审核与待执行验证

```bash
# 只读源代码的静态审计，无torch、无模型。
python qwen_pano/audit_static.py

# 以下CPU测试本次没有运行，由你手动执行；不下载预训练权重。
python -m unittest discover -s qwen_pano/tests -v
```

测试包括上游geometry文件一致性、loss数值和梯度、LR曲线、mask方向、最长caption、house排除、seed算法、随机初始化单层Qwen的LoRA梯度、persistent sampler状态、非默认alpha保存加载。测试通过仍需真实权重的小规模推理与训练来验证环境、内存、有限loss与checkpoint恢复。

训练全景底座不会自动获得因果推理。稳定Qwen-Pano之后，才进入 `idea` 中的world-state labels、GT graph adapter、reasoner与intervention effect/invariance训练。
