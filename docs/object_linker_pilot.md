# Stage 2: region-to-house instance linkage

Run after region_validation.json reports PASS. CPU and NumPy only.

```bash
python -m qwen_pano.caupano.data.matterport.object_linker \
  --regions qwen_pano/outputs/caupano/region_pilot/x8F5xyUWy9e \
  --category-mapping benchmark_assets/Matterport3D_metadata/official/category_mapping.tsv \
  --output qwen_pano/outputs/caupano/object_link_pilot/x8F5xyUWy9e

python -m qwen_pano.caupano.tools.validate_object_links \
  --input qwen_pano/outputs/caupano/object_link_pilot/x8F5xyUWy9e
```

Observed source example: region1 objectId 14 (door) has the same segment set as
house semseg object 4 after adding 1,000,000. This is evidence for a candidate
segment namespace convention, not permission to equate instance numbers. Builder
requires exact set matches for all groups using region_id*1,000,000+local_segment.
It records unmatched groups and never uses nearest-box or category-only fallback.
Validator independently compares segment AABBs and owners against .house E
records, face and object categories, and OBB fields against house semseg JSON.
The assistant has not executed either module.

Outputs: object_links.json, region*_instances.npz, object_link_validation.json.
Reruns replace these generated outputs only. Canonical instance_id is the source
house object ID plus one, unique within a scan; scene identity remains scan_id.
Zero means no assigned instance, not empty geometry. Unannotated faces and unknown
objects retain their geometry. The object table keeps source OBB axes/radii and
the pinned TSV category, without inferring semantic yaw or object-query classes.

House objects without a region-local annotation link remain in the table and are
reported separately. They receive no fabricated face correspondence. This pilot
has 277 house objects, but the user-validated region reports sum to 276 local
groups; the validator reports the difference rather than requiring equal counts.
Do not feed unlinked entries into visible-object supervision by default.

PASS or PASS_WITH_SOURCE_DIFFERENCES allows building a canonical sample index next. Room assignment conflicts,
absent junk-region mesh, semantic validity masks and instance validity masks must
be carried into that index and later dataset code. No ERP rendering or training
dataset completeness is certified by this module.

The user's first builder run exposed source object 24 with category_id=-1;
house semseg independently has label_index=-1. Its segments span several regions.
It is retained as source_category_id=-1, category_mapping_id=-1, mpcat40_id=41
(unlabeled), semantic_valid=false. No region-local instance link is invented.
The validator checks this source case explicitly. After the fix rerun the builder
and then validator in the same output directory; no earlier parser needs rerunning.

The next user run exposed an invalid validator assumption: the containing region
mesh ID need not equal the object region serialized in the raw .house O record.
Six pilot objects have this source difference: 7, 89, 117, 223, 226, 275.
The validator now independently checks exact region/house semseg segment sets
and checks parsed room assignments against raw O records. Segment ownership,
geometry, category and OBB checks remain hard requirements.
Source room differences are reported under source_differences.object_regions,
with both IDs and room_supervision_eligible=false; no source assignment is changed.
The report's room_supervision_excluded_house_object_ids also includes unlinked
objects. The future canonical index must consume this report, not interpret
links[].region_id (mesh partition) as objects[].region_id (source room).
An otherwise successful run reports PASS_WITH_SOURCE_DIFFERENCES and still
ready_for_world_supervision=false. For this validator-only fix, rerun only
validate_object_links; the builder and earlier parsers need not run again.
