"""Validate instance linkage with house E records, geometry and independent OBBs."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from ..data.matterport.camera_parser import member_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.input / 'object_links.json').read_text())
    region_dir = Path(result['region_directory'])
    index = json.loads((region_dir / 'region_index.json').read_text())
    house = json.loads((Path(result['house_directory']) / 'house.json').read_text())
    with ZipFile(Path(result['scan_dir']) / 'house_segmentations.zip') as archive:
        members = member_index(archive)
        name = members[result['scan_id'] + '.semseg.json']
        source_bytes = archive.read(name)
        groups = {r['id']: r for r in json.loads(source_bytes)['segGroups']}
        house_bytes = archive.read(members[result['scan_id'] + '.house'])
        raw_object_regions = {}
        for line in house_bytes.decode().splitlines():
            fields = line.split()
            if fields and fields[0] == 'O':
                raw_object_regions[int(fields[1])] = int(fields[2])
    local_groups = {}
    with ZipFile(Path(result['scan_dir']) / 'region_segmentations.zip') as archive:
        for row in index['regions']:
            for group in json.loads(archive.read(row['semseg_member']))['segGroups']:
                local_groups[(row['region_id'], group['objectId'])] = group
    objects = {r['house_object_id']: r for r in result['objects']}
    source_objects = {r['id']: r for r in house['objects']}
    source_categories = {r['id']: r for r in house['categories']}
    source_segments = {r['mesh_segment_id']: r for r in house['segments']}
    file_by_region = {r['region_id']: r['array_file'] for r in result['region_files']}
    links = {(r['region_id'], r['local_instance_id']): r for r in result['links']}
    errors, region_reports, region_differences = [], [], []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(not result['unresolved_groups'], 'Unresolved region instance groups remain')
    check(hashlib.sha256(source_bytes).hexdigest() == result['house_semseg_sha256'], 'House semseg changed')
    check(result['category_mapping_sha256'] == index['category_mapping_sha256'], 'Category mapping hash mismatch')
    check(len(links) == len(result['links']), 'Duplicate local instance links')
    check(set(links) == set(local_groups), 'Source region instance coverage mismatch')
    check(len({r['house_object_id'] for r in result['links']}) == len(result['links']),
          'Multiple local instances link to the same house object')
    check(set(objects) == set(source_objects), 'House object coverage mismatch')
    max_bbox_error = 0.
    missing_segments = set()
    for row in index['regions']:
        rid = row['region_id']
        with np.load(region_dir / row['array_file'], allow_pickle=False) as archive:
            xyz, faces = archive['vertices_world'], archive['triangles']
            seg, local = archive['face_segment_local'], archive['face_instance_local']
            semantic, raw_category = archive['face_mpcat40'], archive['face_category_mapping_id']
        with np.load(args.input / file_by_region[rid], allow_pickle=False) as archive:
            instance, valid = archive['face_instance_id'], archive['face_instance_valid']
        check(instance.shape == local.shape and valid.shape == local.shape, f'{rid}: face array shape')
        check(np.array_equal(valid, instance > 0), f'{rid}: instance validity mask')
        check(np.all(instance[local < 0] == 0), f'{rid}: unannotated faces assigned invented instance')
        for local_id in np.unique(local[local >= 0]):
            key = (rid, int(local_id))
            if key not in links:
                errors.append(f'{rid}/{local_id}: missing link')
                continue
            link = links[key]
            hid = link['house_object_id']
            obj = objects[hid]
            mask = local == local_id
            check(np.all(instance[mask] == hid + 1), f'{rid}/{local_id}: canonical instance ID mismatch')
            check(link['instance_id'] == hid + 1 and obj['instance_id'] == hid + 1,
                  f'{rid}/{local_id}: object table ID mismatch')
            # Mesh partition and source object-room assignment are separate
            # annotations. Establish identity from exact source segments first.
            source_group = local_groups[key]
            shifted_segments = {rid * result['segment_id_stride'] + s for s in source_group['segments']}
            check(shifted_segments == set(groups[hid]['segments']),
                  f'{rid}/{local_id}: source segment-set linkage mismatch')
            house_region = raw_object_regions[hid]
            if house_region != rid:
                region_differences.append(dict(
                    mesh_region_id=rid, local_instance_id=int(local_id),
                    house_object_id=hid, instance_id=hid + 1,
                    house_region_id=house_region, raw_label=source_group['label'],
                    room_supervision_eligible=False,
                    resolution='pending_room_assignment_review'))
            check(np.all(semantic[mask] == obj['mpcat40_id']), f'{rid}/{local_id}: face/object semantic mismatch')
            check(np.all(raw_category[mask] == obj['category_mapping_id']), f'{rid}/{local_id}: raw category mismatch')

        # Group triangles by segment once; compare each world-space AABB with
        # independently serialized .house E records, including unannotated faces.
        order = np.argsort(seg, kind='stable')
        cuts = np.r_[0, np.flatnonzero(np.diff(seg[order])) + 1, len(order)]
        for start, end in zip(cuts[:-1], cuts[1:]):
            face_ids = order[start:end]
            sid = int(seg[face_ids[0]])
            global_id = rid * result['segment_id_stride'] + sid
            if global_id not in source_segments:
                missing_segments.add(global_id)
                continue
            source = source_segments[global_id]
            points = xyz[faces[face_ids]].reshape(-1, 3)
            actual_box = np.r_[points.min(axis=0), points.max(axis=0)]
            box_error = float(np.max(np.abs(actual_box - np.asarray(source['bbox_world']))))
            max_bbox_error = max(max_bbox_error, box_error)
            check(box_error < 1e-3, f'{rid}/segment {sid}: world bbox differs from house E record')
            ids = np.unique(instance[face_ids][local[face_ids] >= 0])
            check(not len(ids) or (len(ids) == 1 and ids[0] == source['object_id'] + 1),
                  f'{rid}/segment {sid}: house E ownership mismatch')
        region_reports.append(dict(region_id=rid, faces=len(faces),
                                   linked_faces=int(valid.sum()), unassigned_faces=int((~valid).sum())))
    check(not missing_segments, f'{len(missing_segments)} region segments absent from house E records')
    for hid, obj in objects.items():
        source = source_objects[hid]
        group = groups[hid]
        check(source['region_id'] == raw_object_regions[hid],
              f'object {hid}: parsed region differs from raw house O record')
        check(obj['source_category_id'] == source['category_id'], f'object {hid}: source category mismatch')
        if source['category_id'] == -1:
            check(group['label_index'] == obj['category_mapping_id'] == -1,
                  f'object {hid}: unclassified source label mismatch')
            check(obj['mpcat40_id'] == 41 and obj['category'] == 'unlabeled' and not obj['semantic_valid'],
                  f'object {hid}: unclassified object must have no semantic supervision')
        else:
            category = source_categories[source['category_id']]
            check(group['label_index'] == category['category_mapping_id'] == obj['category_mapping_id'],
                  f'object {hid}: house semseg/category mismatch')
        check(obj['instance_id'] == hid + 1 and obj['region_id'] == source['region_id'],
              f'object {hid}: identity/region mismatch')
        for key in ('center_world', 'axes_world_source', 'radii'):
            check(np.array_equal(obj[key], source[key]), f'object {hid}: modified source {key}')
        check(np.allclose(source['center_world'], group['obb']['centroid'], atol=1e-5, rtol=0),
              f'object {hid}: OBB centroid mismatch')
        # Despite its name, MP3D semseg axesLengths stores source radii here.
        check(np.allclose(source['radii'], group['obb']['axesLengths'], atol=1e-5, rtol=0),
              f'object {hid}: OBB radii mismatch')
        axes = np.asarray(group['obb']['normalizedAxes']).reshape(3, 3)
        check(np.allclose(source['axes_world_source'], axes[:2], atol=1e-5, rtol=0),
              f'object {hid}: OBB axes mismatch')
        check(obj['semantic_valid'] == (0 < obj['mpcat40_id'] < 41), f'object {hid}: semantic validity mismatch')
    unlinked = [dict(house_object_id=hid, region_id=objects[hid]['region_id'], category=objects[hid]['category'])
                for hid in result['house_objects_without_region_link']]
    status = 'FAIL' if errors else ('PASS_WITH_SOURCE_DIFFERENCES' if region_differences else 'PASS')
    report = dict(status=status, scan_id=result['scan_id'],
                  linked_local_instances=len(links), house_object_count=len(objects),
                  house_objects_without_region_link=unlinked, regions=region_reports,
                  unclassified_house_object_ids=[hid for hid, obj in objects.items()
                                                 if obj['source_category_id'] == -1],
                  max_segment_bbox_error_m=max_bbox_error,
                  missing_house_segment_ids=sorted(missing_segments), errors=errors,
                  source_differences=dict(object_regions=region_differences),
                  room_supervision_excluded_house_object_ids=sorted(
                      {r['house_object_id'] for r in region_differences}
                      | set(result['house_objects_without_region_link'])),
                  source_house_sha256=hashlib.sha256(house_bytes).hexdigest(),
                  ready_for_canonical_index=not errors, ready_for_world_supervision=False,
                  scope='instance identity and source geometry linkage; no camera/ERP alignment validation')
    (args.input / 'object_link_validation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
