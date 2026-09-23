"""Validate source references and joins before starting ERP alignment work."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--expected-panoramas', type=int, required=True)
    parser.add_argument('--expected-observations', type=int, required=True)
    args = parser.parse_args()
    root = args.input

    def read(path):
        return json.loads(Path(path).read_text())

    manifest = read(root / 'index_manifest.json')
    samples = pq.read_table(root / 'panorama_index.parquet').to_pylist()
    json_rows = [json.loads(line) for line in (root / 'panorama_index.jsonl').read_text().splitlines()]
    house = read(Path(manifest['house_directory']) / 'house.json')
    regions = read(Path(manifest['region_directory']) / 'region_index.json')
    links = read(Path(manifest['object_directory']) / 'object_link_validation.json')
    house_report = read(Path(manifest['house_directory']) / 'house_validation.json')
    camera_path = Path(manifest['camera_directory']) / 'observations.jsonl'
    cameras = [json.loads(line) for line in camera_path.read_text().splitlines()]
    camera_by_id = {r['observation_id']: r for r in cameras}
    panos = {r['panorama_uuid']: r for r in house['panoramas']}
    rooms = {r['id']: r['label'] for r in house['regions']}
    conflicts = {r['panorama_uuid'] for r in house_report['source_differences']['panorama_regions']}
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(samples == json_rows, 'Parquet/JSONL content differs')
    check(len(samples) == args.expected_panoramas == len(panos), 'Panorama count mismatch')
    check({r['panorama_uuid'] for r in samples} == set(panos), 'Panorama coverage mismatch')
    check(len({r['sample_id'] for r in samples}) == len(samples), 'Duplicate sample IDs')
    check(len(cameras) == args.expected_observations, 'Source observation count mismatch')
    for filename, digest in manifest['source_sha256'].items():
        check(hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest,
              f'Upstream changed since index build: {filename}')
    scan_dir = Path(manifest['scan_dir'])
    archive_members = {}
    for filename in ('house_segmentations.zip', 'region_segmentations.zip',
                     'matterport_skybox_images.zip', 'undistorted_color_images.zip',
                     'undistorted_depth_images.zip', 'undistorted_normal_images.zip'):
        with ZipFile(scan_dir / filename) as archive:
            archive_members[str(scan_dir / filename)] = set(archive.namelist())
            if filename == 'house_segmentations.zip':
                member = next(n for n in archive.namelist() if n.endswith('/panorama_to_region.txt'))
                mapping = {p[1]: (int(p[0]), int(p[2]), p[3])
                           for line in archive.read(member).decode().splitlines() if (p := line.split())}

    def check_ref(ref, archive, basename, label):
        archive_path = str(scan_dir / (archive + '.zip'))
        check(ref['archive'] == archive_path, label + ': wrong archive')
        check(ref['member'] in archive_members[archive_path], label + ': ZIP member missing')
        check(Path(ref['member']).name == basename, label + ': wrong source member')

    used = []
    for row in samples:
        uuid = row['panorama_uuid']
        source = panos[uuid]
        check(row['sample_id'] == f"{manifest['scan_id']}_{uuid}" and
              row['scan_id'] == manifest['scan_id'], uuid + ': sample identity')
        check(row['panorama_index'] == source['id'] == mapping[uuid][0], uuid + ': panorama index')
        check(row['region_id'] == source['region_id'], uuid + ': source room changed')
        check(row['region_type'] == rooms.get(source['region_id']), uuid + ': source room type')
        check(row['camera_xyz'] == source['position_world'] and
              row['camera_xyz_source'] == 'house_P_position_world', uuid + ': source position')
        check((row['mapping_region_id'], row['mapping_region_type']) == mapping[uuid][1:],
              uuid + ': mapping source changed')
        check(row['room_assignment_conflict'] == (uuid in conflicts), uuid + ': conflict flag')
        check(row['room_supervision_eligible'] == (source['region_id'] >= 0 and uuid not in conflicts),
              uuid + ': room eligibility')
        check(row['room_supervision_excluded_house_object_ids'] ==
              links['room_supervision_excluded_house_object_ids'], uuid + ': object exclusions lost')
        check(row['split'] is None and row['caption_source'] is None and
              row['erp_camera_to_world'] is None and row['erp_alignment_validated'] is False and
              row['ready_for_training'] is False, uuid + ': premature training/ERP readiness')
        expected_paths = dict(observations_path=camera_path,
                              region_index_path=Path(manifest['region_directory']) / 'region_index.json',
                              object_links_path=Path(manifest['object_directory']) / 'object_links.json',
                              object_validation_path=Path(manifest['object_directory']) / 'object_link_validation.json')
        for key, expected in expected_paths.items():
            check(row[key] == str(expected), uuid + ': ' + key)
        check_ref(row['house_path'], 'house_segmentations', manifest['scan_id'] + '.house', uuid)
        check_ref(row['region_annotation_path'], 'house_segmentations', 'panorama_to_region.txt', uuid)
        check(len(row['mesh_path']) == len(regions['regions']), uuid + ': mesh coverage')
        for ref, region in zip(row['mesh_path'], regions['regions']):
            check_ref(ref, 'region_segmentations', Path(region['mesh_member']).name, uuid)
        check(len(row['rgb_source']) == 6, uuid + ': skybox count')
        for i, ref in enumerate(row['rgb_source']):
            check_ref(ref, 'matterport_skybox_images', f'{uuid}_skybox{i}_sami.jpg', uuid)
        ids = row['observation_ids']
        used.extend(ids)
        expected_ids = {r['observation_id'] for r in cameras if r['panorama_uuid'] == uuid}
        check(set(ids) == expected_ids and len(ids) == len(expected_ids), uuid + ': observation coverage')
        for field in ('perspective_rgb_source', 'depth_source', 'normal_source'):
            check(len(row[field]) == len(ids), uuid + ': ' + field + ' count')
        for i, oid in enumerate(ids):
            camera = camera_by_id[oid]['undistorted']
            check_ref(row['perspective_rgb_source'][i], 'undistorted_color_images', camera['color_file'], oid)
            check_ref(row['depth_source'][i], 'undistorted_depth_images', camera['depth_file'], oid)
            check(len(row['normal_source'][i]) == 3, oid + ': normal component count')
            for ref, axis in zip(row['normal_source'][i], ('nx', 'ny', 'nz')):
                check_ref(ref, 'undistorted_normal_images', Path(camera['depth_file']).stem + '_' + axis + '.png', oid)
    check(Counter(used) == Counter(r['observation_id'] for r in cameras), 'Observation partition mismatch')
    report = dict(status='FAIL' if errors else 'PASS', scan_id=manifest['scan_id'],
                  panorama_count=len(samples), observation_count=len(used),
                  panorama_room_conflict_count=sum(r['room_assignment_conflict'] for r in samples),
                  object_room_conflict_count=len(links['source_differences']['object_regions']),
                  room_supervision_excluded_house_object_ids=links['room_supervision_excluded_house_object_ids'],
                  split_status='unassigned_pilot_not_training_data', errors=errors,
                  ready_for_erp_alignment=not errors, ready_for_training=False,
                  scope='source index and ZIP member references only; no image decoding or ERP geometry validation')
    (root / 'index_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
