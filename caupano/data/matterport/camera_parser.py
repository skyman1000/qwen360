"""Export perspective observation cameras from MP3D ZIPs without extracting images.

Retain source conventions. These are NOT canonical panorama camera poses.
Only Python's standard library is required.
"""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile


def member_index(archive):
    """Use basenames for lookup, but retain exact ZIP names (including //)."""
    return {Path(name).name: name for name in archive.namelist() if not name.endswith('/')}


def matrix(values, width):
    values = [float(value) for value in values]
    return [values[i:i + width] for i in range(0, len(values), width)]


def parse_conf(text):
    header, observations = {}, []
    intrinsics = None
    for line_number, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts or parts[0].startswith('#'):
            continue
        command, values = parts[0], parts[1:]
        if command == 'intrinsics_matrix':
            intrinsics = matrix(values, 3)
        elif command == 'scan':
            if intrinsics is None:
                raise ValueError(f'No intrinsics before scan at line {line_number}')
            observations.append(dict(depth=values[0], color=values[1],
                                     intrinsics=intrinsics,
                                     camera_to_world=matrix(values[2:], 4),
                                     source_line=line_number))
        elif command in ('dataset', 'n_images', 'depth_directory', 'color_directory'):
            header[command] = values[0]
        else:
            raise ValueError(f'Unknown conf command at line {line_number}: {command}')
    if len(observations) != int(header['n_images']):
        raise ValueError('Conf observation count differs from n_images')
    return header, observations


def parse_cameras(scan_dir):
    scan_dir = Path(scan_dir).resolve()
    scan_id = scan_dir.name
    with ZipFile(scan_dir / 'undistorted_camera_parameters.zip') as conf_zip:
        conf_member = member_index(conf_zip)[scan_id + '.conf']
        header, observations = parse_conf(conf_zip.read(conf_member).decode('utf-8'))
    records = []
    with ZipFile(scan_dir / 'matterport_camera_intrinsics.zip') as intr_zip, \
            ZipFile(scan_dir / 'matterport_camera_poses.zip') as pose_zip:
        intr_names, pose_names = member_index(intr_zip), member_index(pose_zip)
        for observation in observations:
            uuid, sensor, yaw = Path(observation['color']).stem.rsplit('_', 2)
            camera_index, yaw_index = int(sensor[1:]), int(yaw)
            intr_member = intr_names[f'{uuid}_intrinsics_{camera_index}.txt']
            pose_member = pose_names[f'{uuid}_pose_{camera_index}_{yaw_index}.txt']
            width, height, fx, fy, cx, cy, *distortion = map(
                float, intr_zip.read(intr_member).decode().split())
            records.append(dict(
                schema_version=1, scan_id=scan_id, panorama_uuid=uuid,
                sample_id=f'{scan_id}_{uuid}',
                observation_id=f'{scan_id}_{uuid}_{camera_index}_{yaw_index}',
                camera_index=camera_index, yaw_index=yaw_index,
                source='gt',
                raw=dict(width=int(width), height=int(height),
                         intrinsics=[[fx, 0., cx], [0., fy, cy], [0., 0., 1.]],
                         distortion=distortion, distortion_order=['k1', 'k2', 'p1', 'p2', 'k3'],
                         camera_to_world=matrix(pose_zip.read(pose_member).decode().split(), 4),
                         camera_frame='x_right_y_down_z_forward', image_origin='top_left',
                         intrinsics_member=intr_member, pose_member=pose_member),
                undistorted=dict(
                    intrinsics=observation['intrinsics'],
                    camera_to_world=observation['camera_to_world'],
                    camera_frame='x_right_y_up_z_backward', image_origin='bottom_left',
                    color_file=observation['color'], depth_file=observation['depth'],
                    conf_member=conf_member, conf_line=observation['source_line']),
            ))
    return sorted(records, key=lambda row: row['observation_id']), header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records, header = parse_cameras(args.scan_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'observations.jsonl').open('w', encoding='utf-8') as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    summary = dict(schema_version=1, scan_id=args.scan_dir.name,
                   scan_dir=str(args.scan_dir.resolve()), conf_header=header,
                   panorama_count=len({r['panorama_uuid'] for r in records}),
                   observation_count=len(records),
                   status='parsed_not_validated',
                   scope='perspective_cameras_only; panorama pose and ERP alignment pending')
    (args.output / 'parse_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
