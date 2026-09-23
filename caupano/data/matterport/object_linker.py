"""Link region-local instances to house objects using exact source segment sets.

The region*1_000_000+local segment convention is a candidate observed in the
pilot sources. Every group must match the house semseg set exactly; no nearest
object or category-only matching is used. Run validate_object_links afterwards.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .camera_parser import member_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regions', type=Path, required=True)
    parser.add_argument('--category-mapping', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.regions / 'region_index.json').read_text())
    validation = json.loads((args.regions / 'region_validation.json').read_text())
    if validation['status'] != 'PASS' or validation['scan_id'] != index['scan_id']:
        raise ValueError('Region validation must PASS for this scan first')
    house_path = Path(index['house_directory']) / 'house.json'
    house = json.loads(house_path.read_text())
    mapping_hash = hashlib.sha256(args.category_mapping.read_bytes()).hexdigest()
    if mapping_hash != index['category_mapping_sha256']:
        raise ValueError('Use the pinned category mapping from region parsing')
    with args.category_mapping.open() as handle:
        category_map = {int(r['index']): r for r in csv.DictReader(handle, delimiter='\t')}
    categories = {r['id']: r for r in house['categories']}
    source_objects = {r['id']: r for r in house['objects']}
    scan_dir = Path(index['scan_dir'])
    with ZipFile(scan_dir / 'house_segmentations.zip') as archive:
        member = member_index(archive)[house['scan_id'] + '.semseg.json']
        semseg_bytes = archive.read(member)
    house_groups = json.loads(semseg_bytes)['segGroups']
    by_segments = {}
    for group in house_groups:
        key = frozenset(group['segments'])
        if key in by_segments:
            raise ValueError('Duplicate house object segment sets; cannot uniquely link')
        by_segments[key] = group
    args.output.mkdir(parents=True, exist_ok=True)
    links, issues, region_files = [], [], []
    with ZipFile(scan_dir / 'region_segmentations.zip') as archive:
        for region in index['regions']:
            rid = region['region_id']
            groups = json.loads(archive.read(region['semseg_member']))['segGroups']
            local_to_house = {}
            for group in groups:
                local_segments = group['segments']
                if not local_segments or min(local_segments) < 0 or max(local_segments) >= 1_000_000:
                    raise ValueError(f'Region {rid}: local segments outside inspected ID convention')
                key = frozenset(rid * 1_000_000 + s for s in local_segments)
                target = by_segments.get(key)
                if target is None:
                    issues.append(dict(region_id=rid, local_instance_id=group['objectId'],
                                       reason='no exact house semseg segment-set match'))
                    continue
                hid = target['id']
                if hid not in source_objects:
                    raise ValueError(f'House semseg object {hid} absent from house object table')
                local_to_house[group['objectId']] = hid
                links.append(dict(region_id=rid, local_instance_id=group['objectId'],
                                  house_object_id=hid, instance_id=hid + 1,
                                  raw_label=group['label'], segment_count=len(key),
                                  method='exact_segment_set', source='derived'))
            with np.load(args.regions / region['array_file'], allow_pickle=False) as arrays:
                local_instances = arrays['face_instance_local']
            canonical = np.zeros(local_instances.shape, dtype=np.int32)
            for local_id, house_id in local_to_house.items():
                canonical[local_instances == local_id] = house_id + 1
            filename = f'region{rid}_instances.npz'
            np.savez_compressed(args.output / filename, face_instance_id=canonical,
                                face_instance_valid=canonical > 0)
            region_files.append(dict(region_id=rid, array_file=filename))
    linked_ids = {r['house_object_id'] for r in links}
    objects = []
    for obj in house['objects']:
        if obj['category_id'] == -1:
            # MP3D explicitly stores unclassified objects (pilot object 24).
            # Keep the source identity and geometry, but provide no class target.
            mapping_id, mpcat40_id, category_name = -1, 41, 'unlabeled'
        else:
            category = categories[obj['category_id']]
            mapping_id = category['category_mapping_id']
            mapped = category_map[mapping_id]
            mpcat40_id, category_name = int(mapped['mpcat40index']), mapped['mpcat40']
        objects.append(dict(instance_id=obj['id'] + 1, house_object_id=obj['id'],
                            region_id=obj['region_id'], source_category_id=obj['category_id'],
                            category_mapping_id=mapping_id,
                            mpcat40_id=mpcat40_id, category=category_name,
                            semantic_valid=0 < mpcat40_id < 41,
                            center_world=obj['center_world'], axes_world_source=obj['axes_world_source'],
                            radii=obj['radii'], geometry_source='gt',
                            has_region_instance_link=obj['id'] in linked_ids))
    result = dict(schema_version=1, scan_id=house['scan_id'], scan_dir=str(scan_dir),
                  region_directory=str(args.regions.resolve()), house_directory=index['house_directory'],
                  status='built_not_validated', segment_id_stride=1_000_000,
                  instance_id_rule='house_object_id + 1; zero = no assigned instance, not empty geometry',
                  category_mapping_sha256=mapping_hash,
                  house_semseg_sha256=hashlib.sha256(semseg_bytes).hexdigest(),
                  region_assignment_conflicts=index['region_assignment_conflicts'],
                  links=links, objects=objects, region_files=region_files, unresolved_groups=issues,
                  house_objects_without_region_link=sorted(set(source_objects) - linked_ids))
    (args.output / 'object_links.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], linked_local_instances=len(links),
                          linked_house_objects=len(linked_ids), house_object_count=len(objects),
                          unresolved_groups=issues,
                          unclassified_house_object_ids=[r['house_object_id'] for r in objects
                                                         if r['source_category_id'] == -1],
                          house_objects_without_region_link=result['house_objects_without_region_link']), indent=2))


if __name__ == '__main__':
    main()
