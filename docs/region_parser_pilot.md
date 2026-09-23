# Stage 2: region-local parsing

House parsing passed with recorded source differences. This module intentionally
stops before region-local to house-global object linkage. In inspected examples,
local instance IDs differ from house object indices, and local segment numbers
differ from house mesh segment IDs. Never join them by integer equality.

Run from DIT360 in the existing qwen360 environment (NumPy, no GPU):

```bash
python -m qwen_pano.caupano.data.matterport.region_parser \
  --house qwen_pano/outputs/caupano/house_pilot/x8F5xyUWy9e \
  --category-mapping benchmark_assets/Matterport3D_metadata/official/category_mapping.tsv \
  --output qwen_pano/outputs/caupano/region_pilot/x8F5xyUWy9e

python -m qwen_pano.caupano.tools.validate_regions \
  --input qwen_pano/outputs/caupano/region_pilot/x8F5xyUWy9e \
  --category-mapping benchmark_assets/Matterport3D_metadata/official/category_mapping.tsv
```

Parser reads one region at a time from the ZIP and writes region*.npz plus
region_index.json. It supports the inspected little-endian triangular PLY schema
only; no generic PLY framework is introduced. Reruns replace these output files.
Validator writes region_validation.json. Neither command has been run by the
assistant; source inspection is not a successful module validation.

NPZ fields:
- vertices_world: N x 3; triangles: F x 3.
- face_segment_local: PLY material_id, compared against fsegs.segIndices.
- face_instance_local: PLY segment_id, compared against semseg objectId.
- face_category_mapping_id: PLY category_id, referring to TSV index.
- face_mpcat40: pinned TSV mapping, not legacy .house mpcat40.
- face_semantic_valid: false for void (0) or unlabeled (41).

Instance 0 is a valid LOCAL object. This is not canonical instance ERP, which
reserves 0 for background. Face semantic validity also does not imply that a
category is furniture: floor/wall/ceiling still need layout handling downstream.

Pilot archive contains meshes for regions 0..13. House region 14 is labeled Z
(junk) and has no mesh; this is reported, not synthesized or deleted. Missing
non-junk meshes remain errors. The 3 conflicting room assignments from house
validation are carried forward and are not resolved by region parsing.

Only after validation should global object linkage, canonical index and geometric
projection be implemented. Pure parser PASS is not certification of training data.
