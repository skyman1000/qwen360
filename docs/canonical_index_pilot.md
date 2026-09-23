# Stage 2: source-level Canonical Sample Index

This implements the detailed v3 section 12 index for one pilot scan. No ERP
rendering, model training, caption generation or world-state construction occurs.
Keep pilot output separate from the eventual multi-scan data/canonical dataset.

Requires PyArrow for Parquet. Check availability with `python -c "import pyarrow"`.
If missing, install only the new dependency: `python -m pip install --no-deps pyarrow`.
No existing training dependency needs upgrading.

From the DIT360 root:

```bash
python -m qwen_pano.caupano.data.canonical_index \
  --cameras qwen_pano/outputs/caupano/camera_pilot/x8F5xyUWy9e \
  --objects qwen_pano/outputs/caupano/object_link_pilot/x8F5xyUWy9e \
  --output qwen_pano/outputs/caupano/canonical_pilot/x8F5xyUWy9e

python -m qwen_pano.caupano.tools.validate_canonical_index \
  --input qwen_pano/outputs/caupano/canonical_pilot/x8F5xyUWy9e \
  --expected-panoramas 43 \
  --expected-observations 774
```

Outputs: panorama_index.parquet, matching panorama_index.jsonl,
index_manifest.json, index_validation.json. Expected counts: 43 panorama samples,
774 perspective observations, 3 panorama-room source conflicts, 6 object-room
source conflicts. Excluded house object IDs: 7, 24, 89, 117, 223, 226, 275.
The assistant has not executed these modules; user validation is still required.

## Field semantics at this stage

- IDs follow the detailed plan. region_id/type retain the house annotation;
  region_type is the original case-sensitive letter, not a guessed English name.
  mapping_region_id/type preserve panorama_to_region.txt separately.
- camera_xyz retains the house P position in world coordinates. It is not a
  substitute for an ERP camera-to-world matrix; that field is null until alignment.
- ZIP references contain archive and exact member separately, preserving double
  slashes in original member names. No extraction is needed.
- rgb_source lists six source skybox faces in filename index order, with no
  assumed face orientation. perspective_rgb_source/depth_source/normal_source
  follow observation_ids order. Normals list nx, ny, nz components per observation;
  boundary/confidence auxiliary PNGs are not normal vector components.
- mesh_path references all available region meshes for the scan. Room assignment
  is not used to infer visibility or omit geometry. Object tables remain scan-level
  external references, not a claimed visible-object list per panorama.
- Panorama room-conflict flags and object room-supervision exclusions propagate
  into every relevant sample. Object semantic/instance masks remain in referenced
  validated object/region files; no masks are replaced with all-valid defaults.
- split and caption_source remain null. A scan-level split manifest is required
  before producing training data; never assign independent panorama splits.
- ERP pose is null, erp_alignment_validated and ready_for_training are false.

Validator PASS certifies index joins, source membership, unchanged upstream JSON
hashes and propagation of existing conflicts. It does not resolve those conflicts
or certify image contents, visibility, ERP alignment, or complete training data.
Next module after user validation: panorama pose / skybox-to-ERP alignment pilot.
