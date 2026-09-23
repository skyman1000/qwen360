# First train-scan construction

Local predecessor train.npy contains sT4fr6TAbpF (44 listed panorama records).
Its local skybox ZIP already exists; full-panorama caption files are available.
The raw house may contain more panoramas than the filtered predecessor list;
prepare_scan uses source house counts, not 44 as a hardcoded expected count.
The batch manifest inherits scan-level train membership for all raw panoramas
and records missing captions. This defines a scan-expanded construction set,
not an exact reproduction of the predecessor's panorama subset.

Download using the existing download_mp_py3.py and its normal terms prompt:

```bash
python download_mp_py3.py \
  -o benchmark_assets/Matterport3D_raw --id sT4fr6TAbpF \
  --type matterport_skybox_images matterport_camera_intrinsics matterport_camera_poses \
  undistorted_camera_parameters undistorted_color_images undistorted_depth_images \
  undistorted_normal_images house_segmentations region_segmentations
```

Then run in the existing CPU geometry environment:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
qwen_pano/outputs/caupano/geometry_env/bin/python \
  -m qwen_pano.caupano.tools.prepare_scan \
  --scan-dir benchmark_assets/Matterport3D_raw/v1/scans/sT4fr6TAbpF \
  --work-root qwen_pano/outputs/caupano --threads 4
```

No parser/validator needs a separate user command. Existing parsers still enforce
source consistency; an unsupported source case stops preparation with the stage
log, rather than manufacturing a successful index. A completed index is reused
for an unchanged scan. Batch jobs resume existing panorama reports as documented
in scan_dataset_batch.md. The assistant has not run this new orchestrator.

The current test-scan result is 40 assembled / 43 requested. Three samples are
retained as NEEDS_REVIEW due to source-camera sensor/mesh depth disagreement or
no valid comparison pixels; do not delete or mark them passed. This is a filtered
pilot subset, not the full benchmark. Report selection and coverage for any
later evaluation. Resolving these three does not block preparing train scans.

One train house is an integration/overfit pilot, not sufficient evidence of
cross-house generalization. Add train houses after this run; keep test houses
out of training. Next model work is dataset loading plus Object/Relation/Layout
Encoders and World Adapter, preserving the original text condition.

## Correction: source panorama split crosses one house

The earlier designation of sT4fr6TAbpF as a train house was incomplete and wrong.
Full inspection shows 44 source train panoramas and 37 source test panoramas in
this house. Across the local metadata, panorama overlap is zero, but this one
house overlaps. PanFusion's non-layout loader consumes each npy panorama row;
its README attributes these lists to MVDiffusion. These are not a verified
house-disjoint Matterport benchmark split.

Continue construction with `--split-policy defer-conflicts` on prepare_scan or
build_scan_dataset. This sets split=null for the conflicting house, stores
source_scan_splits and source_panorama_splits, and leaves training readiness
false. It does not assign the house to train or test. The earlier instruction
that the batch would inherit train for this house is superseded by this correction.
Previously validated index outputs can be reused; no downloads/parsers must repeat.
A future explicit house-level manifest must resolve the split before training.
