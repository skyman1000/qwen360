"""Build a source-level panorama index; ERP alignment and training remain pending."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

from .matterport.camera_parser import member_index


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cameras', type=Path, required=True)
    parser.add_argument('--objects', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    object_dir = args.objects.resolve()
    camera_dir = args.cameras.resolve()
    links = read_json(object_dir / 'object_links.json')
    link_report = read_json(object_dir / 'object_link_validation.json')
    house_dir = Path(links['house_directory'])
    region_dir = Path(links['region_directory'])
    house = read_json(house_dir / 'house.json')
    regions = read_json(region_dir / 'region_index.json')
    scan_dir = Path(links['scan_dir'])
    scan_id = links['scan_id']
    reports = [read_json(camera_dir / 'camera_validation.json'),
               read_json(house_dir / 'house_validation.json'),
               read_json(region_dir / 'region_validation.json'), link_report]
    for report in reports:
        if report['scan_id'] != scan_id or report['errors']:
            raise ValueError('Upstream validation must pass for the same scan')
    if not link_report['ready_for_canonical_index']:
        raise ValueError('Object linkage is not ready for canonical indexing')
    camera_summary = read_json(camera_dir / 'parse_summary.json')
    if Path(camera_summary['scan_dir']).resolve() != scan_dir.resolve():
        raise ValueError('Camera and object source directories differ')
    observations = defaultdict(list)
    for line in (camera_dir / 'observations.jsonl').read_text().splitlines():
        row = json.loads(line)
        observations[row['panorama_uuid']].append(row)
    if set(observations) != {p['panorama_uuid'] for p in house['panoramas']}:
        raise ValueError('Camera and house panorama coverage differs')

    archive_names = ['matterport_skybox_images', 'undistorted_color_images',
                     'undistorted_depth_images', 'undistorted_normal_images',
                     'house_segmentations', 'region_segmentations']
    members = {}
    for name in archive_names:
        with ZipFile(scan_dir / (name + '.zip')) as archive:
            members[name] = member_index(archive)

    def ref(archive, basename):
        return dict(archive=str(scan_dir / (archive + '.zip')), member=members[archive][basename])

    with ZipFile(scan_dir / 'house_segmentations.zip') as archive:
        mapping = {}
        for line in archive.read(members['house_segmentations']['panorama_to_region.txt']).decode().splitlines():
            index, uuid, region, label = line.split()
            mapping[uuid] = dict(index=int(index), region=int(region), label=label)
    room_types = {r['id']: r['label'] for r in house['regions']}
    conflicts = {r['panorama_uuid']: r for r in reports[1]['source_differences']['panorama_regions']}
    mesh_refs = [ref('region_segmentations', Path(r['mesh_member']).name) for r in regions['regions']]
    samples = []
    for pano in sorted(house['panoramas'], key=lambda r: r['id']):
        uuid, rid = pano['panorama_uuid'], pano['region_id']
        obs = sorted(observations[uuid], key=lambda r: (r['camera_index'], r['yaw_index']))
        depth, normal, color = [], [], []
        for row in obs:
            source = row['undistorted']
            color.append(ref('undistorted_color_images', source['color_file']))
            depth.append(ref('undistorted_depth_images', source['depth_file']))
            stem = Path(source['depth_file']).stem
            normal.append([ref('undistorted_normal_images', stem + '_' + axis + '.png')
                           for axis in ('nx', 'ny', 'nz')])
        samples.append(dict(
            schema_version=1, sample_id=f'{scan_id}_{uuid}', scan_id=scan_id,
            panorama_uuid=uuid, panorama_index=pano['id'],
            region_id=rid, region_type=room_types[rid] if rid >= 0 else None,
            region_type_encoding='source_house_region_label',
            mapping_region_id=mapping[uuid]['region'], mapping_region_type=mapping[uuid]['label'],
            room_assignment_conflict=uuid in conflicts,
            room_supervision_eligible=rid >= 0 and uuid not in conflicts,
            camera_xyz=pano['position_world'], camera_xyz_source='house_P_position_world',
            split=None, caption_source=None,
            rgb_source=[ref('matterport_skybox_images', f'{uuid}_skybox{i}_sami.jpg') for i in range(6)],
            rgb_source_kind='source_skybox_faces_0_to_5_not_aligned_erp',
            perspective_rgb_source=color, depth_source=depth, normal_source=normal,
            depth_source_kind='undistorted_perspective_not_erp',
            normal_source_kind='undistorted_perspective_xyz_components_not_erp',
            observation_ids=[r['observation_id'] for r in obs],
            observations_path=str(camera_dir / 'observations.jsonl'),
            mesh_path=mesh_refs, mesh_scope='all_available_scan_regions_not_visibility_filtered',
            house_path=ref('house_segmentations', scan_id + '.house'),
            region_annotation_path=ref('house_segmentations', 'panorama_to_region.txt'),
            region_index_path=str(region_dir / 'region_index.json'),
            object_links_path=str(object_dir / 'object_links.json'),
            object_validation_path=str(object_dir / 'object_link_validation.json'),
            room_supervision_excluded_house_object_ids=link_report['room_supervision_excluded_house_object_ids'],
            erp_camera_to_world=None, erp_alignment_validated=False, ready_for_training=False))

    source_files = [camera_dir / 'observations.jsonl', camera_dir / 'parse_summary.json',
                    camera_dir / 'camera_validation.json', house_dir / 'house.json',
                    house_dir / 'house_validation.json', region_dir / 'region_index.json',
                    region_dir / 'region_validation.json', object_dir / 'object_links.json',
                    object_dir / 'object_link_validation.json']
    manifest = dict(schema_version=1, scan_id=scan_id, scan_dir=str(scan_dir),
                    camera_directory=str(camera_dir), house_directory=str(house_dir),
                    region_directory=str(region_dir), object_directory=str(object_dir),
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                    status='built_not_validated', panorama_count=len(samples),
                    observation_count=sum(len(r['observation_ids']) for r in samples),
                    panorama_room_conflict_count=len(conflicts),
                    split_status='unassigned_pilot_not_training_data',
                    ready_for_training=False)
    args.output.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(samples), args.output / 'panorama_index.parquet')
    (args.output / 'panorama_index.jsonl').write_text(
        ''.join(json.dumps(r, allow_nan=False) + '\n' for r in samples))
    (args.output / 'index_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
