# Structured World State construction

Consumes the completed geometry pilot and validated source index. No parser,
image alignment or full-scene geometry generation is repeated. Run from DIT360:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
qwen_pano/outputs/caupano/geometry_env/bin/python \
  -m qwen_pano.caupano.data.world_state_builder \
  --index qwen_pano/outputs/caupano/canonical_pilot/x8F5xyUWy9e \
  --geometry qwen_pano/outputs/caupano/geometry_pilot/x8F5xyUWy9e/03de84eb12c24a93bdfd88e46a6db25a \
  --output qwen_pano/outputs/caupano/world_state_pilot/x8F5xyUWy9e/03de84eb12c24a93bdfd88e46a6db25a \
  --threads 4
```

The assistant has not executed this new module. It uses the existing geometry
environment and CPU ray casting. Output is world_state.json, objects.json,
relations.json, camera.json, layout.json, visibility.npz and build_report.json.
Image assets remain referenced in the geometry directory; they are not copied.

## Implemented fields

- All source objects remain in objects.json with complete OBB axes, radii,
  size=2*radii, centres, camera-frame spherical positions, category and masks.
- World-state objects are the visible, semantically valid linked objects with
  at least 16 visible pixels. This threshold is explicit, not an existence label.
  Invisible/unknown objects are not labelled absent. No fixed object-token cap
  is introduced at this stage.
- For each linked object, cast the same ERP rays at its isolated mesh, using
  a conservative mesh-AABB prefilter. Projected area is the number of rays hitting
  that isolated object, not OBB projected area. Visibility is full-scene visible
  pixels / isolated-mesh projected pixels. A zero denominator is invalid, not
  full invisibility. Unlinked object 24 retains null area and ratio.
- Room extent/type retain source annotation. Type is explicitly a Matterport
  letter code; it is not silently expanded to an unverified English category.
- Layout uses source floor polygons in vertex-ID order, median floor vertex z,
  and median ceiling semantic-hit z within the room footprint. Missing ceiling
  evidence stays null, not a fabricated room-bbox ceiling. This is coarse layout,
  not a watertight reconstructed room.
- Room containment requires the assigned room polygon and vertical extent;
  known room conflicts remain excluded. Centre containment does not imply the
  entire OBB lies inside the room.
- Support uses OBB bottom/top gap <=8cm, vertical-axis compatibility and object
  footprint overlap >=25% of supported-object footprint. Floor is separate.
  These are derived heuristic relations, not manual GT; confidence is explicitly
  uncalibrated. Support inverse can be obtained from supported_by edges.
- Other spatial relations use up to four nearest neighbours per object. Near
  uses 15% of primary room diagonal, clamped to [0.5,2] meters. Above/below use
  separated OBB vertical extents. Left/right use wrapped longitude with 15-degree
  dead zones around zero and antipodes. Front/behind means camera +z centre order
  with a 25cm margin, not a semantic object-facing relation. Frame and method are
  recorded on every relation. These thresholds need dataset-level tuning later.

## Minimal schema clarification

The detailed plan's yaw_world field cannot be a semantic forward heading from
the available OBB alone. Here it is the longest near-horizontal OBB axis azimuth
modulo pi, or null if none. yaw_semantic_valid is always false, and the provenance
is derived. Size still follows original OBB axis order. An encoder must honour
the yaw mask; it must not equate size[0] to semantic forward length.

The camera matrix maps the left-handed ERP coordinate basis to source world
coordinates, so determinant -1 is expected here. It is not an SO(3) rotation for
quaternion conversion. Camera alignment is derived; source object geometry is
GT. Prior pilot evidence supports construction, not a blanket dataset-wide GT
certification. Upstream candidate reports remain unchanged.

Build summary reports visibility inconsistencies rather than clipping ratios.
There is no separate validator command. ready_for_training remains false because
scan splits, caption linkage, multi-panorama coverage and model integration are
not yet complete, not because another per-file review command is required.
