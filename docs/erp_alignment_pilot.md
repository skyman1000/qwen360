# One-command RGB and orientation diagnostic

Run from DIT360 root in qwen360. Uses CPU only (four Torch/OpenCV threads),
NumPy, Pillow, Torch, OpenCV with SIFT, SciPy and the existing bundled PanFusion
converter. Does not load diffusion weights or change any training data/config.
The assistant has not run this module; user execution and visual review remain.

```bash
python -m qwen_pano.caupano.tools.erp_alignment_pilot \
  --index qwen_pano/outputs/caupano/canonical_pilot/x8F5xyUWy9e \
  --panorama-uuid 03de84eb12c24a93bdfd88e46a6db25a \
  --output qwen_pano/outputs/caupano/erp_alignment_pilot/x8F5xyUWy9e/03de84eb12c24a93bdfd88e46a6db25a
```

Outputs rgb_erp.png (2048x1024), alignment_report.json and, when enough matches
exist to estimate a basis, alignment_contact_sheet.jpg. Its 18 rows contain
source perspective / ERP reprojection / overlay. FIT and HELD OUT label which
observations contributed to estimation. Compare major edges and distant objects
especially on HELD OUT rows. Small local parallax/exposure differences are possible.

CANDIDATE_REQUIRES_VISUAL_REVIEW means at least 60 fit inliers, 30 held-out
inliers distributed over at least three held-out observations. It is not PASS
for GT world alignment. WEAK_MATCH_REQUIRES_REVIEW or INSUFFICIENT_MATCHES
leaves the candidate unapproved; do not lower thresholds to claim success.
All statuses retain erp_alignment_validated=false and ready_for_training=false.
No canonical-index fields or source files are overwritten. Rerunning replaces
pilot output files. Provide the report and contact sheet for the next review.

See coordinate_convention.md for the legacy endpoint-grid difference, camera
axes and handedness. Confirm image-based orientation first; metric depth/mesh
alignment belongs to the next data-generation unit, not this visual estimate.
