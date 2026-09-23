# Stage 2: house parser pilot

Run only after camera validation passes. From the DIT360 workspace:

```bash
python -m qwen_pano.caupano.data.matterport.house_parser \
  --scan-dir benchmark_assets/Matterport3D_raw/v1/scans/x8F5xyUWy9e \
  --output qwen_pano/outputs/caupano/house_pilot/x8F5xyUWy9e

python -m qwen_pano.caupano.tools.validate_house \
  --input qwen_pano/outputs/caupano/house_pilot/x8F5xyUWy9e \
  --cameras qwen_pano/outputs/caupano/camera_pilot/x8F5xyUWy9e \
  --category-mapping benchmark_assets/Matterport3D_metadata/official/category_mapping.tsv
```

Parser uses the standard library; validator uses NumPy, CPU only. Nothing is
extracted and no model is loaded. Outputs are house.json, parse_summary.json and
house_validation.json. Reruns replace these generated files.

The inspected house header declares 774 images, 43 panoramas, 104 vertices,
15 surfaces, 19788 segments, 277 objects, 1659 categories, 15 regions, 0 portals,
and 1 level. These are expected source counts, not a completed test result.

Source format and implementation:
- https://github.com/niessner/Matterport/blob/master/data_organization.md
- https://github.com/niessner/Matterport/blob/master/code/gaps/apps/mpview/mp.cpp

Important interface decisions:
- House image extrinsics are world-to-camera, inverse to the undistorted .conf
  camera-to-world pose. The validator compares independently supplied matrices.
- Source indices remain zero-based; -1 means unassigned, not object/background 0.
  Canonical instance IDs will require an explicit mapping in the later builder.
- Both portals and panoramas use P records, with 15 versus 13 fields in ASCII 1.1.
  This pilot has zero portals; portal parsing remains unvalidated on real records.
- Preserve region labels as codes and source surface labels. Do not infer
  complete walls/ceilings from floor-only surface annotations.
- Preserve panorama positions from house records, not averages of sensor poses.
- Preserve both source OBB axes and radii. Canonical size, normalized third axis,
  and yaw belong to the later object builder. The first OBB axis can be vertical;
  it is not guaranteed to denote semantic object heading.
- Degenerate OBBs and panoramas without region assignments are reported, not
  silently removed. Training inclusion rules are a later dataset decision.
- No split assignment, World State, normal frame, ERP origin or visibility is
  inferred. PASS certifies the checks listed in house_validation.json only.

Next, after user validation: region mesh/segment/object label parsing, then
canonical index and coordinate/ERP projection QC. No later module is implemented
as part of this house parser step.

## Source differences found in the pilot

The initial user-run validation found three panoramas assigned to regions in
the .house but marked -1 in panorama_to_region.txt, and ten category mappings
differing between the .house and the downloaded official category_mapping.tsv.
These differences exist in the original sources, not only in parsed JSON.
Their historical cause/version ordering has not been established.

The revised validator first checks parsed panorama/category fields against the
original .house lines. Parse errors, invalid IDs and geometric failures remain
errors. Cross-file label differences are then recorded with both values,
affected object IDs and source hashes. They produce PASS_WITH_SOURCE_DIFFERENCES
with exit code 0 only when no validation errors remain. This permits region
parsing; it does not certify labels as ready for supervision.

house.json remains unchanged. Conflicted panorama room assignments are marked
ineligible for room supervision in the report pending geometric review; no
panorama is deleted. The later canonical builder must consume these flags and
apply a single pinned category mapping consistently to objects and mesh labels,
preserving source labels and treating unlabeled as unknown, not as a furniture
class. This report does not itself implement those downstream transformations.

Only rerun validate_house after this change; camera and house parsing need not
be repeated.
