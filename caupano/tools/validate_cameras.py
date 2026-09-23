"""CPU-only, cross-source validation of exported MP3D cameras; requires NumPy.

Checks camera geometry, source coverage and image references. Does not certify
RGB/depth registration, panorama orientation or rendered geometry.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import struct
from zipfile import ZipFile

import numpy as np

from ..data.matterport.camera_parser import member_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Camera parser output directory')
    parser.add_argument('--expected-panoramas', type=int, required=True)
    parser.add_argument('--expected-observations', type=int, required=True)
    args = parser.parse_args()
    summary = json.loads((args.input / 'parse_summary.json').read_text())
    rows = [json.loads(line) for line in (args.input / 'observations.jsonl').read_text().splitlines()]
    scan_dir = Path(summary['scan_dir'])
    errors, groups = [], defaultdict(list)
    maxima = dict(rotation_orthogonality=0., rotation_determinant=0.,
                  raw_conf_pose=0., raw_conf_intrinsics=0.)

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(len(rows) == args.expected_observations, 'Unexpected observation count')
    check(len({r['observation_id'] for r in rows}) == len(rows), 'Duplicate observation IDs')
    basis_change = np.diag([1., -1., -1., 1.])
    for row in rows:
        oid = row['observation_id']
        groups[row['panorama_uuid']].append(row)
        for source in ('raw', 'undistorted'):
            pose = np.asarray(row[source]['camera_to_world'])
            intrinsic = np.asarray(row[source]['intrinsics'])
            check(pose.shape == (4, 4) and intrinsic.shape == (3, 3), f'{oid}: matrix shape')
            check(np.isfinite(pose).all() and np.isfinite(intrinsic).all(), f'{oid}: nonfinite matrix')
            rotation = pose[:3, :3]
            orth_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
            det_error = float(abs(np.linalg.det(rotation) - 1.))
            maxima['rotation_orthogonality'] = max(maxima['rotation_orthogonality'], orth_error)
            maxima['rotation_determinant'] = max(maxima['rotation_determinant'], det_error)
            check(orth_error < 1e-4 and det_error < 1e-4, f'{oid}/{source}: invalid rotation')
            check(np.allclose(pose[3], [0, 0, 0, 1]), f'{oid}/{source}: homogeneous row')
            check(intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0, f'{oid}/{source}: focal length')
        raw, und = row['raw'], row['undistorted']
        pose_error = float(np.max(np.abs(np.asarray(raw['camera_to_world']) @ basis_change
                                        - np.asarray(und['camera_to_world']))))
        maxima['raw_conf_pose'] = max(maxima['raw_conf_pose'], pose_error)
        check(pose_error < 1e-4, f'{oid}: raw/conf frame conversion mismatch')
        # Cross-check the pixel-origin change against independently supplied K.
        expected_intrinsics = np.array(raw['intrinsics'])
        expected_intrinsics[1, 2] = raw['height'] - 1 - expected_intrinsics[1, 2]
        intr_error = float(np.max(np.abs(expected_intrinsics - np.asarray(und['intrinsics']))))
        maxima['raw_conf_intrinsics'] = max(maxima['raw_conf_intrinsics'], intr_error)
        check(intr_error < .01, f'{oid}: raw/conf intrinsic correspondence mismatch')
        expected_stem = f"{row['panorama_uuid']}_d{row['camera_index']}_{row['yaw_index']}"
        check(Path(und['depth_file']).stem == expected_stem, f'{oid}: depth identity mismatch')

    check(len(groups) == args.expected_panoramas, 'Unexpected panorama count')
    expected_views = {(camera, yaw) for camera in range(3) for yaw in range(6)}
    for uuid, observations in groups.items():
        check(len(observations) == 18 and
              {(r['camera_index'], r['yaw_index']) for r in observations} == expected_views,
              f'{uuid}: expected 3 sensors x 6 yaw observations')

    with ZipFile(scan_dir / 'matterport_camera_poses.zip') as archive:
        expected = {f"{r['panorama_uuid']}_pose_{r['camera_index']}_{r['yaw_index']}.txt" for r in rows}
        check(set(member_index(archive)) == expected, 'Raw pose source coverage mismatch')
    with ZipFile(scan_dir / 'matterport_camera_intrinsics.zip') as archive:
        expected = {f"{r['panorama_uuid']}_intrinsics_{r['camera_index']}.txt" for r in rows}
        check(set(member_index(archive)) == expected, 'Raw intrinsics source coverage mismatch')

    packages = ['undistorted_color_images', 'undistorted_depth_images',
                'undistorted_normal_images', 'matterport_skybox_images']
    for package in packages:
        with ZipFile(scan_dir / (package + '.zip')) as archive:
            names = member_index(archive)
            for row in rows:
                und = row['undistorted']
                if package == 'undistorted_color_images':
                    required = [und['color_file']]
                elif package == 'undistorted_depth_images':
                    required = [und['depth_file']]
                elif package == 'undistorted_normal_images':
                    stem = Path(und['depth_file']).stem
                    required = [f'{stem}_{axis}.png' for axis in ('nx', 'ny', 'nz')]
                else:
                    if (row['camera_index'], row['yaw_index']) != (0, 0):
                        continue
                    required = [f"{row['panorama_uuid']}_skybox{i}_sami.jpg" for i in range(6)]
                for name in required:
                    check(name in names, f'{package}: missing {name}')
            # Read only the PNG header of one depth image, not the full image archive.
            if package == 'undistorted_depth_images' and rows:
                first = rows[0]
                with archive.open(names[first['undistorted']['depth_file']]) as handle:
                    header = handle.read(29)
                width, height = struct.unpack('>II', header[16:24])
                check(header[:8] == b'\x89PNG\r\n\x1a\n' and header[24] == 16,
                      'Expected 16-bit depth PNG')
                check((width, height) == (first['raw']['width'], first['raw']['height']),
                      'Pilot depth dimensions differ from raw camera dimensions')

    report = dict(status='PASS' if not errors else 'FAIL', scan_id=summary['scan_id'],
                  panorama_count=len(groups), observation_count=len(rows),
                  max_errors=maxima, errors=errors,
                  scope='camera records and source correspondence only; ERP alignment not yet validated')
    destination = args.input / 'camera_validation.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
