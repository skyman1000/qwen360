# Stage 2: camera parser pilot

Run from the DIT360 parent workspace. No GPU, model loading, archive extraction,
or changes to Qwen-Pano training are involved. Parser uses the standard library;
validator additionally uses NumPy. Output is under the existing ignored outputs/.

```bash
python -m qwen_pano.caupano.data.matterport.camera_parser \
  --scan-dir benchmark_assets/Matterport3D_raw/v1/scans/x8F5xyUWy9e \
  --output qwen_pano/outputs/caupano/camera_pilot/x8F5xyUWy9e

python -m qwen_pano.caupano.tools.validate_cameras \
  --input qwen_pano/outputs/caupano/camera_pilot/x8F5xyUWy9e \
  --expected-panoramas 43 --expected-observations 774
```

Parser writes observations.jsonl and parse_summary.json. Validator writes
camera_validation.json and exits nonzero on failed checks. Rerunning replaces
these generated reports. Expected counts come from the inspected source .house
header and .conf, not from a completed validation run.

## Source conventions and scope

Source: https://github.com/niessner/Matterport/blob/master/data_organization.md

- Raw camera: x right, y down, z forward; top-left image coordinates.
- Undistorted .conf camera: x right, y up, z backward; bottom-left image coordinates.
- Both source poses are camera-to-world, row-major serialized, acting on column
  vectors, with translations in meters. They are retained separately.
- The sampled source pair follows T_conf = T_raw @ diag(1,-1,-1,1) and
  cy_conf = H-1-cy_raw. The validator checks these relations across all observations.
  The later depth projection module must explicitly convert stored image row
  indices to the source convention. No image flipping is performed here.
- Actual pilot filenames use camera indices 0..2 and yaw indices 0..5. The
  official prose contains contradictory ranges; source records take precedence.
- Observation camera centers are not collapsed into a panorama center. A later
  house parser must supply panorama identity/position and region membership.
- No ERP pose, yaw origin, split, or normal-vector frame is inferred here.
- No RGB-depth registration or rendered geometry validation is claimed by PASS.

Each JSONL row contains scan/sample/observation IDs, camera/yaw indices, raw
intrinsics/distortion/pose, undistorted intrinsics/pose/image filenames, and exact
ZIP member/line provenance. The .conf intrinsic matrix applies to subsequent
scan lines until replaced. Raw dimensions remain under raw, rather than silently
being assigned to all undistorted images.

After the user runs validation successfully: implement house parser, validate
panorama/region/OBB records, then region parser and canonical index. Only after
coordinate and RGB alignment checks proceed to depth/normal/semantic projection.
