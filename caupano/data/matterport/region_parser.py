"""Read region-local MP3D mesh/face labels; no global object IDs are inferred."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .camera_parser import member_index


def read_region_ply(data):
    """Read the exact binary triangle schema inspected in the pilot release."""
    stream = io.BytesIO(data)
    header = []
    while True:
        line = stream.readline().decode('ascii').strip()
        if not line:
            raise ValueError('Unexpected end of PLY header')
        header.append(line)
        if line == 'end_header':
            break
    if header[:2] != ['ply', 'format binary_little_endian 1.0']:
        raise ValueError('Expected binary_little_endian PLY')
    vi = next(i for i, line in enumerate(header) if line.startswith('element vertex '))
    fi = next(i for i, line in enumerate(header) if line.startswith('element face '))
    nv, nf = int(header[vi].split()[-1]), int(header[fi].split()[-1])
    vertex_properties = [f'property float {p}' for p in ('x', 'y', 'z', 'nx', 'ny', 'nz', 'tx', 'ty')]
    vertex_properties += [f'property uchar {p}' for p in ('red', 'green', 'blue')]
    face_properties = ['property list uchar int vertex_indices', 'property int material_id',
                       'property int segment_id', 'property int category_id']
    if header[vi + 1:fi] != vertex_properties or header[fi + 1:-1] != face_properties:
        raise ValueError('Uninspected PLY schema: retain source and inspect before extending reader')
    vertex_dtype = np.dtype([('values', '<f4', (8,)), ('rgb', 'u1', (3,))])
    face_dtype = np.dtype([('count', 'u1'), ('vertices', '<i4', (3,)),
                          ('material_id', '<i4'), ('segment_id', '<i4'), ('category_id', '<i4')])
    vertices = np.frombuffer(data, dtype=vertex_dtype, count=nv, offset=stream.tell())
    offset = stream.tell() + nv * vertex_dtype.itemsize
    faces = np.frombuffer(data, dtype=face_dtype, count=nf, offset=offset)
    if len(data) != offset + nf * face_dtype.itemsize or not np.all(faces['count'] == 3):
        raise ValueError('Expected exactly the declared number of triangular faces')
    return dict(vertices_world=vertices['values'][:, :3].copy(), triangles=faces['vertices'].copy(),
                face_segment_local=faces['material_id'].copy(),
                face_instance_local=faces['segment_id'].copy(),
                face_category_mapping_id=faces['category_id'].copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--house', type=Path, required=True, help='Validated house output directory')
    parser.add_argument('--category-mapping', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    house = json.loads((args.house / 'house.json').read_text())
    report = json.loads((args.house / 'house_validation.json').read_text())
    if report['scan_id'] != house['scan_id'] or not report['ready_for_region_parser']:
        raise ValueError('House validation must permit region parsing for the same scan')
    category_bytes = args.category_mapping.read_bytes()
    mapping_hash = hashlib.sha256(category_bytes).hexdigest()
    if mapping_hash != report['source_sha256']['category_mapping']:
        raise ValueError('Use the category mapping already recorded by house validation')
    mapping = {int(r['index']): r for r in csv.DictReader(io.StringIO(category_bytes.decode()), delimiter='\t')}
    scan_dir = Path(house['scan_dir'])
    args.output.mkdir(parents=True, exist_ok=True)
    regions = []
    with ZipFile(scan_dir / 'region_segmentations.zip') as archive:
        names = member_index(archive)
        mesh_names = sorted((n for n in names if n.endswith('.ply')), key=lambda n: int(n[6:-4]))
        for name in mesh_names:
            region_id = int(name[6:-4])
            stem = name[:-4]
            source_data = archive.read(names[name])
            arrays = read_region_ply(source_data)
            # Canonical vocabulary uses this pinned TSV, never old house mpcat40.
            raw_categories = arrays['face_category_mapping_id']
            canonical = np.full(raw_categories.shape, 41, dtype=np.int32)
            for category_id in np.unique(raw_categories):
                if category_id == 0:
                    canonical[raw_categories == category_id] = 0  # void
                elif category_id > 0:
                    canonical[raw_categories == category_id] = int(mapping[int(category_id)]['mpcat40index'])
            arrays['face_mpcat40'] = canonical
            arrays['face_semantic_valid'] = (canonical > 0) & (canonical < 41)
            np.savez_compressed(args.output / (stem + '.npz'), **arrays)
            regions.append(dict(region_id=region_id, mesh_member=names[name],
                                fsegs_member=names[stem + '.fsegs.json'],
                                semseg_member=names[stem + '.semseg.json'],
                                mesh_sha256=hashlib.sha256(source_data).hexdigest(),
                                vertex_count=len(arrays['vertices_world']), face_count=len(arrays['triangles']),
                                array_file=stem + '.npz'))
    present = {r['region_id'] for r in regions}
    summary = dict(schema_version=1, scan_id=house['scan_id'], scan_dir=str(scan_dir),
                   house_directory=str(args.house.resolve()), category_mapping_sha256=mapping_hash,
                   status='parsed_not_validated', region_mesh_count=len(regions), regions=regions,
                   house_regions_without_mesh=[dict(region_id=r['id'], label=r['label'])
                                               for r in house['regions'] if r['id'] not in present],
                   region_assignment_conflicts=report['source_differences']['panorama_regions'],
                   scope='region-local faces and labels; house object linkage and ERP projection pending')
    (args.output / 'region_index.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'regions'}, indent=2))


if __name__ == '__main__':
    main()
