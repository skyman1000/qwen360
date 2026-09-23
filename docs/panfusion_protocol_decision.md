# CauPano pilot: source-panorama protocol and frozen epoch005

The user reopened the AI-generated house-exclusive requirement and authorized a
small-fit experiment using pano_official_full_59263/checkpoint-epoch005. For this
pilot, recommend source PanFusion/MVDiffusion panorama assignments. House-exclusive
splitting remains a separate cross-house evaluation option, not a requirement
for testing explicit world-conditioning on the source protocol.

PanFusion README attributes its splits to MVDiffusion:
https://github.com/chengzhag/PanFusion/blob/main/README.md
The local loader dataset/Matterport3D.py consumes each train.npy/test.npy panorama.
Local metadata has 9820 train and 1092 test entries, no panorama overlap and one
shared house (sT4fr6TAbpF: 44 train / 37 test). The 67 assembled samples from that
house break down into 36 train / 31 test. The 40 assembled x8F5xyUWy9e samples
are source test. Keep geometry filtering and excluded-sample coverage explicit.

Do not rerun parsers, geometry, visibility or world-state construction. Use
export_panfusion_protocol to write new manifest-only joins. Original metadata,
pilot reports, null-split manifests and in-progress Qwen-Pano training remain intact.
The export gives 36 train / 71 test from currently assembled data, plus a seeded
8-sample fit and 4-sample dev drawn only from source train. These are pilot subsets,
not the full source benchmark. Model-side loaders must take split/caption from
the experiment manifest, not the unassigned original world_state JSON.

The frozen epoch005 checkpoint is complete, but its training_config records
split_status=official_full_training_only_no_independent_test_claim. Its base LoRA
training manifest contains this pilot's buildings, including source test imagery.
The exporter records exact (scan, panorama) exposure counts. A later adapter-only
split cannot erase that exposure. This checkpoint is suitable for fixed-baseline
engineering/control comparisons, not proof of unseen-image/generalization gains.
No need to retrain or stop the current Qwen-Pano run for this pilot.

GT-world-conditioned evaluation must be distinguished from text-only evaluation:
test GT world state supplies scene information by design in Stage 3. The final
Text-to-World pipeline must predict that condition from text without test GT.
Report GT-world control and end-to-end text-only generation separately.

Use the same epoch005, prompts, seeds and sampling settings for adapter/no-adapter
comparisons. Train only new encoders and world adapter. Avoid global source object
IDs as learned features; keep them for indexing relations, as IDs could memorize
objects shared across panoramas. Fit/dev from one house test integration, not
cross-house generalization. World-conditioned trainer integration is still pending;
exporting manifests does not make an existing LoRA trainer world-conditioned.
