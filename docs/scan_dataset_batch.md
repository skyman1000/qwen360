# Batch construction of the existing pilot scan

The existing test.npy assigns all 43 x8F5xyUWy9e panoramas to test. Preserve that
scan-level split. Do not turn these pilot samples into a formal training dataset.
This batch module does not alter current Qwen-Pano training or original pilot files.

Run in the existing geometry environment:

```bash
qwen_pano/outputs/caupano/geometry_env/bin/python \
  -m qwen_pano.caupano.tools.build_scan_dataset \
  --index qwen_pano/outputs/caupano/canonical_pilot/x8F5xyUWy9e \
  --work-root qwen_pano/outputs/caupano \
  --threads 4
```

Sequentially runs existing alignment, geometry and world-state builders, logs
stdout/stderr per sample/stage and continues other samples after a stage failure.
It reuses existing report files without re-running completed stages. Reused reports
must have the expected sample ID and accepted status; a rejected report is listed
as NEEDS_REVIEW, never silently promoted. This is a simple resume mechanism for
unchanged inputs/config; if inputs or builder definitions change, use a new
work-root instead of trusting old reports. Geometry height is fixed to 512.
The new orchestrator has not been executed by the assistant.

Output: outputs/caupano/scan_dataset/x8F5xyUWy9e/{samples.jsonl,batch_report.json,logs/}.
Each assembled record references RGB and world_state, and joins the existing
full-panorama caption and source scan split. The external world-state pilot file
retains null split/caption; the dataset manifest is the authoritative join.
Natural captions do not imply exact pose/relation supervision; those masks stay
false until a separate text-to-world target policy is implemented.

ASSEMBLED is not visual approval or model-readiness. Inspect failure summaries and
sample a few multi-room overlays after the batch, rather than pausing after each
panorama. This implementation prioritizes reuse and processes CPU jobs sequentially;
it rebuilds the ray-casting scene per panorama and is not yet optimized for all
Matterport houses. No elapsed-time guarantee is made.

After this pilot scan: acquire complete raw fields for selected train scans,
reuse the same parsers/index/builders for them, and preserve held-out scans.
Then implement the model-side dataset loader and Encoder/Adapter interfaces.
