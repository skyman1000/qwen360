"""Read MP3D ASCII 1.1 house records, preserving source IDs and geometry."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from .camera_parser import matrix, member_index


TABLES = ['images', 'panoramas', 'vertices', 'surfaces', 'segments',
          'objects', 'categories', 'regions', 'portals', 'levels']


def floats(values):
    return [float(value) for value in values]


def parse_house(scan_dir):
    scan_dir = Path(scan_dir).resolve()
    with ZipFile(scan_dir / 'house_segmentations.zip') as archive:
        member = member_index(archive)[scan_dir.name + '.house']
        lines = archive.read(member).decode().splitlines()
    if lines[0].split() != ['ASCII', '1.1']:
        raise ValueError('This parser supports the inspected ASCII 1.1 house format')
    house = dict(schema_version=1, scan_id=scan_dir.name, scan_dir=str(scan_dir),
                 source='gt', source_member=member,
                 scope='source house records; not canonical world state',
                 **{table: [] for table in TABLES})
    for line_number, line in enumerate(lines[1:], 2):
        p = line.split()
        if not p:
            continue
        tag = p[0]
        row = dict(source_line=line_number)
        if tag == 'H':
            house['declared_counts'] = dict(zip(TABLES, map(int, p[3:13])))
            house['bbox_world'] = floats(p[18:24])
            continue
        if tag == 'L':
            table = 'levels'
            row.update(id=int(p[1]), label=p[3], position_world=floats(p[4:7]),
                       bbox_world=floats(p[7:13]))
        elif tag == 'R':
            table = 'regions'
            row.update(id=int(p[1]), level_id=int(p[2]), label=p[5],
                       position_world=floats(p[6:9]), bbox_world=floats(p[9:15]),
                       height=float(p[15]))
        elif tag == 'P' and len(p) == 15:
            table = 'portals'
            row.update(id=int(p[1]), region_ids=[int(p[2]), int(p[3])], label=p[4],
                       endpoints_world=[floats(p[5:8]), floats(p[8:11])])
        elif tag == 'P' and len(p) == 13:
            table = 'panoramas'
            row.update(id=int(p[2]), panorama_uuid=p[1], region_id=int(p[3]),
                       sample_id=f'{scan_dir.name}_{p[1]}', position_world=floats(p[5:8]))
        elif tag == 'S':
            table = 'surfaces'
            row.update(id=int(p[1]), region_id=int(p[2]), label=p[4],
                       position_world=floats(p[5:8]), normal_world=floats(p[8:11]),
                       bbox_world=floats(p[11:17]))
        elif tag == 'V':
            table = 'vertices'
            row.update(id=int(p[1]), surface_id=int(p[2]), label=p[3],
                       position_world=floats(p[4:7]), normal_world=floats(p[7:10]))
        elif tag == 'I':
            table = 'images'
            row.update(id=int(p[1]), panorama_id=int(p[2]), panorama_uuid=p[3],
                       camera_index=int(p[4]), yaw_index=int(p[5]),
                       observation_id=f'{scan_dir.name}_{p[3]}_{p[4]}_{p[5]}',
                       world_to_camera=matrix(p[6:22], 4), intrinsics=matrix(p[22:31], 3),
                       camera_frame='x_right_y_up_z_backward', image_origin='bottom_left',
                       width=int(p[31]), height=int(p[32]), position_world=floats(p[33:36]))
        elif tag == 'C':
            table = 'categories'
            row.update(id=int(p[1]), category_mapping_id=int(p[2]),
                       category_mapping_name=p[3].replace('#', ' '),
                       mpcat40_id=int(p[4]), mpcat40_name=p[5].replace('#', ' '))
        elif tag == 'O':
            table = 'objects'
            row.update(id=int(p[1]), region_id=int(p[2]), category_id=int(p[3]),
                       center_world=floats(p[4:7]), axes_world_source=matrix(p[7:13], 3),
                       radii=floats(p[13:16]))
        elif tag == 'E':
            table = 'segments'
            row.update(id=int(p[1]), object_id=int(p[2]), mesh_segment_id=int(p[3]),
                       area=float(p[4]), position_world=floats(p[5:8]), bbox_world=floats(p[8:14]))
        else:
            raise ValueError(f'Unsupported record at line {line_number}: {line}')
        house[table].append(row)
    return house


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    house = parse_house(args.scan_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'house.json').write_text(json.dumps(house, allow_nan=False) + '\n')
    summary = dict(scan_id=house['scan_id'], status='parsed_not_validated',
                   declared_counts=house['declared_counts'],
                   parsed_counts={table: len(house[table]) for table in TABLES},
                   panoramas_without_region=sum(p['region_id'] == -1 for p in house['panoramas']),
                   note='Source IDs remain zero-based; -1 denotes an unassigned reference')
    (args.output / 'parse_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
