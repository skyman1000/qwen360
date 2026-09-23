"""Cross-check a parsed house against camera observations and official metadata.

CPU only. No geometry rendering or canonical world-state construction.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from ..data.matterport.camera_parser import member_index
from ..data.matterport.house_parser import TABLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--cameras', type=Path, required=True, help='Validated camera output directory')
    parser.add_argument('--category-mapping', type=Path, required=True)
    args = parser.parse_args()
    house = json.loads((args.input / 'house.json').read_text())
    camera_report = json.loads((args.cameras / 'camera_validation.json').read_text())
    if camera_report['status'] != 'PASS' or camera_report['scan_id'] != house['scan_id']:
        raise ValueError('Camera validation must PASS for the same scan first')
    cameras = {r['observation_id']: r for r in
               map(json.loads, (args.cameras / 'observations.jsonl').read_text().splitlines())}
    with args.category_mapping.open() as handle:
        category_map = {int(r['index']): r for r in csv.DictReader(handle, delimiter='\t')}
    errors = []
    source_differences = dict(panorama_regions=[], category_mapping=[])
    tables = {name: {r['id']: r for r in house[name]} for name in TABLES}

    def check(condition, message):
        if not condition:
            errors.append(message)

    def reference(value, table, description):
        check(value == -1 or value in tables[table], f'{description}: unknown {table} id {value}')

    for name in TABLES:
        check(len(house[name]) == house['declared_counts'][name], f'{name}: header count mismatch')
        check(len(tables[name]) == len(house[name]), f'{name}: duplicate IDs')
        for row in house[name]:
            for field in ('position_world', 'center_world', 'bbox_world', 'normal_world', 'height', 'area'):
                if field in row:
                    check(np.isfinite(row[field]).all(), f'{name}/{row["id"]}: nonfinite {field}')
            if 'bbox_world' in row:
                box = row['bbox_world']
                check(all(a <= b for a, b in zip(box[:3], box[3:])), f'{name}/{row["id"]}: inverted bbox')
    for row in house['regions']:
        reference(row['level_id'], 'levels', f'region {row["id"]}')
    for row in house['surfaces']:
        reference(row['region_id'], 'regions', f'surface {row["id"]}')
    for row in house['vertices']:
        reference(row['surface_id'], 'surfaces', f'vertex {row["id"]}')
    for row in house['portals']:
        for region in row['region_ids']:
            reference(region, 'regions', f'portal {row["id"]}')

    panoramas = {r['panorama_uuid']: r for r in house['panoramas']}
    check(len(panoramas) == len(house['panoramas']), 'Duplicate panorama UUIDs')
    check(set(panoramas) == {r['panorama_uuid'] for r in cameras.values()}, 'House/camera panorama coverage')
    for row in house['panoramas']:
        reference(row['region_id'], 'regions', f'panorama {row["id"]}')
    with ZipFile(Path(house['scan_dir']) / 'house_segmentations.zip') as archive:
        member = member_index(archive)['panorama_to_region.txt']
        mapping_bytes = archive.read(member)
        house_bytes = archive.read(house['source_member'])
        mapping = [line.split() for line in mapping_bytes.decode().splitlines() if line.strip()]
    # Establish parser fidelity before treating cross-file discrepancies as source
    # differences. Never downgrade an incorrectly parsed house value to a warning.
    source_lines = house_bytes.decode().splitlines()
    for row in house['panoramas']:
        p = source_lines[row['source_line'] - 1].split()
        check(p[0] == 'P' and
              (row['panorama_uuid'], row['id'], row['region_id']) == (p[1], int(p[2]), int(p[3])),
              f'panorama {row["id"]}: parsed identity/region differs from source house')
    for row in house['categories']:
        p = source_lines[row['source_line'] - 1].split()
        check(p[0] == 'C' and
              (row['id'], row['category_mapping_id'], row['mpcat40_id'], row['mpcat40_name']) ==
              (int(p[1]), int(p[2]), int(p[4]), p[5].replace('#', ' ')),
              f'category {row["id"]}: parsed category differs from source house')
    check(len(mapping) == len(panoramas), 'panorama_to_region row count')
    check({r[1] for r in mapping} == set(panoramas), 'panorama_to_region UUID coverage')
    for index, uuid, region, label in mapping:
        row = panoramas[uuid]
        check(row['id'] == int(index), f'{uuid}: panorama index mismatch')
        reference(int(region), 'regions', f'{uuid}/panorama_to_region')
        if row['region_id'] != int(region):
            source_differences['panorama_regions'].append(dict(
                panorama_uuid=uuid, house_region_id=row['region_id'],
                mapping_region_id=int(region), mapping_label=label,
                room_supervision_eligible=False,
                action='preserve both sources; defer room supervision until geometry review'))
        expected_label = '-' if int(region) == -1 else tables['regions'][int(region)]['label']
        check(label == expected_label, f'{uuid}: region label mismatch')

    maxima = dict(house_conf_inverse=0., image_camera_position_m=0., intrinsics=0., obb_axes=0.)
    check({r['observation_id'] for r in house['images']} == set(cameras), 'House/camera image coverage')
    for row in house['images']:
        oid = row['observation_id']
        camera = cameras[oid]
        reference(row['panorama_id'], 'panoramas', oid)
        check(tables['panoramas'][row['panorama_id']]['panorama_uuid'] == row['panorama_uuid'],
              f'{oid}: panorama parent mismatch')
        c2w = np.asarray(camera['undistorted']['camera_to_world'])
        inverse_error = float(np.max(np.abs(np.asarray(row['world_to_camera']) @ c2w - np.eye(4))))
        position_error = float(np.max(np.abs(np.asarray(row['position_world']) - c2w[:3, 3])))
        intrinsic_error = float(np.max(np.abs(np.asarray(row['intrinsics']) - camera['undistorted']['intrinsics'])))
        maxima['house_conf_inverse'] = max(maxima['house_conf_inverse'], inverse_error)
        maxima['image_camera_position_m'] = max(maxima['image_camera_position_m'], position_error)
        maxima['intrinsics'] = max(maxima['intrinsics'], intrinsic_error)
        # Decimal text serialization of poses and inverse poses introduces rounding.
        check(inverse_error < 1e-3, f'{oid}: house extrinsics is not inverse of conf pose')
        check(position_error < 1e-4, f'{oid}: camera position mismatch')
        check(intrinsic_error < .01, f'{oid}: intrinsics mismatch')
        check((row['width'], row['height']) == (camera['raw']['width'], camera['raw']['height']),
              f'{oid}: camera dimensions mismatch')

    for row in house['categories']:
        mapped = category_map[row['category_mapping_id']]
        if row['mpcat40_id'] != int(mapped['mpcat40index']):
            source_differences['category_mapping'].append(dict(
                house_category_id=row['id'], category_mapping_id=row['category_mapping_id'],
                raw_category=mapped['raw_category'],
                house_mpcat40_id=row['mpcat40_id'], house_mpcat40_name=row['mpcat40_name'],
                tsv_mpcat40_id=int(mapped['mpcat40index']), tsv_mpcat40_name=mapped['mpcat40'],
                affected_object_ids=[obj['id'] for obj in house['objects'] if obj['category_id'] == row['id']],
                action='preserve house labels; canonical builder must use one pinned mapping consistently'))
    degenerate = []
    for row in house['objects']:
        reference(row['region_id'], 'regions', f'object {row["id"]}')
        reference(row['category_id'], 'categories', f'object {row["id"]}')
        axes, radii = np.asarray(row['axes_world_source']), np.asarray(row['radii'])
        check(np.isfinite(axes).all() and np.isfinite(radii).all(), f'object {row["id"]}: nonfinite OBB')
        axis_error = float(np.max(np.abs(axes @ axes.T - np.eye(2))))
        maxima['obb_axes'] = max(maxima['obb_axes'], axis_error)
        check(axis_error < 1e-4, f'object {row["id"]}: invalid OBB axes')
        check((radii >= 0).all(), f'object {row["id"]}: negative OBB radius')
        if (radii <= 1e-6).any():
            degenerate.append(row['id'])
    for row in house['segments']:
        reference(row['object_id'], 'objects', f'segment {row["id"]}')
    check(len({r['mesh_segment_id'] for r in house['segments']}) == len(house['segments']),
          'Duplicate mesh segment IDs')

    has_differences = any(source_differences.values())
    status = 'FAIL' if errors else ('PASS_WITH_SOURCE_DIFFERENCES' if has_differences else 'PASS')
    report = dict(status=status, scan_id=house['scan_id'],
                  counts={name: len(house[name]) for name in TABLES}, max_errors=maxima,
                  panoramas_without_region=[r['panorama_uuid'] for r in house['panoramas'] if r['region_id'] == -1],
                  degenerate_obb_object_ids=degenerate,
                  surface_labels=dict(Counter(r['label'] for r in house['surfaces'])),
                  source_differences=source_differences,
                  source_difference_counts={key: len(value) for key, value in source_differences.items()},
                  source_sha256=dict(house=hashlib.sha256(house_bytes).hexdigest(),
                                     panorama_to_region=hashlib.sha256(mapping_bytes).hexdigest(),
                                     category_mapping=hashlib.sha256(args.category_mapping.read_bytes()).hexdigest()),
                  ready_for_region_parser=not errors,
                  ready_for_world_supervision=False,
                  errors=errors,
                  scope='house records and cross-source associations; no ERP, yaw, visibility or mesh projection validation')
    (args.input / 'house_validation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
