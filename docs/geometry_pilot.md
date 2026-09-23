# Stage 2: one-command geometric ERP asset generation

This module consumes the existing index, camera records, region arrays, global
instance links, and supported image-alignment candidate. It does not repeat those
parsers or alter their outputs. Geometry is generated at 1024x512 by default;
this is a pilot resolution, not a change to Qwen-Pano training resolution.

## Isolated CPU dependency

Open3D 0.19 supports Python 3.12 and CPU ray casting. Install into a venv inheriting
read access to the current qwen360 packages; any new dependency installations
stay in that venv. Do not activate it or install Open3D into the training env.

```bash
python -m venv --system-site-packages qwen_pano/outputs/caupano/geometry_env
qwen_pano/outputs/caupano/geometry_env/bin/python -m pip install 'open3d-cpu==0.19.0'

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
qwen_pano/outputs/caupano/geometry_env/bin/python \
  -m qwen_pano.caupano.data.build_geometry_pilot \
  --index qwen_pano/outputs/caupano/canonical_pilot/x8F5xyUWy9e \
  --alignment qwen_pano/outputs/caupano/erp_alignment_pilot/x8F5xyUWy9e/03de84eb12c24a93bdfd88e46a6db25a \
  --output qwen_pano/outputs/caupano/geometry_pilot/x8F5xyUWy9e/03de84eb12c24a93bdfd88e46a6db25a \
  --height 512 --threads 4
```

The assistant has not executed the new module or installed its dependency.
No diffusion model or GPU is used. The complete region mesh scene is loaded in
CPU RAM; it includes all 14 available regions rather than filtering by room ID.

## Sources and conventions

- [Matterport source format](https://github.com/niessner/Matterport/blob/master/data_organization.md):
  depth is axial, divide PNG values by 4000 for meters; zero has no measurement.
- Conf intrinsics have bottom-left image origin and x-right/y-up/z-backward frame.
  PNG row y maps to conf vertical coordinate H-1-y. Depth backprojection is
  ((x-cx)/fx, (H-1-y-cy)/fy, -1) times axial depth, then conf camera-to-world.
- Sensor points use each observation's own translation, not a shared centre.
  The candidate panorama origin is only used for final ERP radial distance.
- Pixel-centred ERP bins keep the nearest sensor point; holes stay invalid.
  RGB accompanies the same winning sensor point, enabling RGB alignment review.
- Mesh unit rays yield radial depth and nearest triangle. Semantic/instance
  arrays are gathered from already validated face labels and preserve masks.
- Mesh normals are transformed into ERP coordinates, normalized, and oriented
  toward the camera. This is the detailed plan's mesh-normal fallback; official
  sensor-normal transformation has not been validated and is not mixed in.
- Legacy RGB is resampled into a new pixel-centred rgb_erp.png in this output
  directory. No existing training RGB or alignment-pilot RGB is overwritten.

## Outputs

- rgb_erp.png, rgb_sensor_projection.png
- depth_sensor_erp.npy, depth_sensor_valid.npy
- depth_mesh_erp.npy, depth_mesh_valid.npy
- normal_erp.npy (H,W,3), normal_valid.npy
- semantic_erp.npy (pinned mpcat40), semantic_valid.npy
- instance_erp.npy (house object ID + 1), instance_valid.npy
- layout_erp.npy, layout_valid.npy, layout.json
- visibility.json (all source objects, visible pixels and solid angle)
- geometry_overview.jpg, geometry_report.json

Depth units are meters; invalid depth/normal values are zero with explicit masks.
Instance zero means unassigned, not empty geometry. Layout is only a semantic
mesh-derived dense coarse label map, not a closed room model or room polygon.

Visibility's projected_pixel_count and visibility_ratio remain null until the
object-area module is implemented. A zero visible-pixel count is not evidence
that an object does not exist. Visibility is conditional on the candidate pose.
Room exclusions remain in the geometry report; no source room ID is repaired.

## Integrated checks and interpretation

The same command checks nonempty depths, overlap, normal lengths and instance IDs,
reports ERP sensor/mesh depth errors, and compares sparse source-camera depth
against mesh ray casts without using the ERP basis. That source-camera check
helps distinguish a raw camera/depth issue from an ERP RGB orientation issue.
Semantic overlays and projected sensor RGB provide visual alignment evidence.

BUILT_FOR_REVIEW means assets were produced with internal checks passing, not GT
pose approval. Review flags use a 10% median relative depth error and 25% sensor
coverage as screening heuristics. They do not change masks or delete difficult
pixels. Errors near transparent water/glass, occlusion edges and mesh holes can
be real source differences. Do not lower thresholds solely to clear a flag.

The candidate basis and origin are recorded but never written back as validated
GT; ready_for_training remains false. After this integrated result is reviewed,
continue to object-area/visibility, structured layout and spatial relations as
one World State construction unit, rather than rerunning the earlier parsers.
