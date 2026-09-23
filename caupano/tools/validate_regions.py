"""Check region-local face labels against independent fsegs/semseg annotations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--category-mapping', type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.input / 'region_index.json').read_text())
    house = json.loads((Path(index['house_directory']) / 'house.json').read_text())
    with args.category_mapping.open() as handle:
        mapping = {int(r['index']): r for r in csv.DictReader(handle, delimiter='\t')}
    errors, summaries = [], []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(hashlib.sha256(args.category_mapping.read_bytes()).hexdigest() == index['category_mapping_sha256'],
          'Category mapping changed since parsing')
    house_regions = {r['id'] for r in house['regions']}
    check(len({r['region_id'] for r in index['regions']}) == len(index['regions']), 'Duplicate region IDs')
    for missing in index['house_regions_without_mesh']:
        check(missing['label'] == 'Z', f"Non-junk region lacks mesh: {missing['region_id']}")
    with ZipFile(Path(index['scan_dir']) / 'region_segmentations.zip') as archive:
        for row in index['regions']:
            region_id = row['region_id']
            check(region_id in house_regions, f'{region_id}: absent from house regions')
            with np.load(args.input / row['array_file'], allow_pickle=False) as archive_arrays:
                arrays = {k: archive_arrays[k] for k in archive_arrays.files}
            xyz, triangles = arrays['vertices_world'], arrays['triangles']
            segments, instances = arrays['face_segment_local'], arrays['face_instance_local']
            categories = arrays['face_category_mapping_id']
            nf = len(triangles)
            check(len(xyz) == row['vertex_count'] and nf == row['face_count'], f'{region_id}: count mismatch')
            check(np.isfinite(xyz).all(), f'{region_id}: nonfinite vertices')
            check(triangles.shape == (nf, 3) and ((triangles >= 0) & (triangles < len(xyz))).all(),
                  f'{region_id}: invalid triangle indices')
            for key in ('face_segment_local', 'face_instance_local', 'face_category_mapping_id',
                        'face_mpcat40', 'face_semantic_valid'):
                check(arrays[key].shape == (nf,), f'{region_id}: wrong face array shape {key}')
            fsegs = json.loads(archive.read(row['fsegs_member']))
            check(fsegs['elementType'] == 'faces' and np.array_equal(segments, fsegs['segIndices']),
                  f'{region_id}: PLY material_id differs from fsegs face segments')
            groups = json.loads(archive.read(row['semseg_member']))['segGroups']
            owners, labels = {}, {}
            for group in groups:
                oid = int(group['objectId'])
                check(oid not in labels, f'{region_id}: duplicate local object ID {oid}')
                labels[oid] = group['label']
                for seg in group['segments']:
                    check(seg not in owners, f'{region_id}: segment {seg} belongs to multiple groups')
                    owners[seg] = oid
            expected_instances = np.array([owners.get(int(s), -1) for s in segments], dtype=np.int32)
            annotated = expected_instances >= 0
            check(np.array_equal(instances[annotated], expected_instances[annotated]),
                  f'{region_id}: PLY instance differs from semseg ownership')
            # Do not silently accept a labeled mesh instance absent from JSON groups.
            check(np.all(instances[~annotated] < 0), f'{region_id}: PLY labels faces absent from semseg')
            for oid, label in labels.items():
                raw_ids = np.unique(categories[instances == oid])
                for raw_id in raw_ids:
                    if raw_id > 0:
                        check(mapping[int(raw_id)]['raw_category'].replace('#', ' ') == label.replace('#', ' '),
                              f'{region_id}/object {oid}: mesh category differs from semseg label')
            for raw_id in np.unique(categories):
                expected = 41 if raw_id < 0 else (0 if raw_id == 0 else int(mapping[int(raw_id)]['mpcat40index']))
                check(np.all(arrays['face_mpcat40'][categories == raw_id] == expected),
                      f'{region_id}: canonical category mapping mismatch {raw_id}')
            check(np.array_equal(arrays['face_semantic_valid'],
                                 (arrays['face_mpcat40'] > 0) & (arrays['face_mpcat40'] < 41)),
                  f'{region_id}: semantic validity mask mismatch')
            summaries.append(dict(region_id=region_id, vertices=len(xyz), faces=nf,
                                  local_object_groups=len(groups), unannotated_faces=int((~annotated).sum()),
                                  unknown_or_void_faces=int((~arrays['face_semantic_valid']).sum())))
    report = dict(status='FAIL' if errors else 'PASS', scan_id=index['scan_id'],
                  region_mesh_count=len(index['regions']), regions=summaries,
                  house_regions_without_mesh=index['house_regions_without_mesh'],
                  inherited_room_assignment_conflicts=len(index['region_assignment_conflicts']),
                  ready_for_global_object_linkage=not errors, ready_for_world_supervision=False,
                  errors=errors, scope='region-local mesh and label consistency only; no global instance mapping or ERP QC')
    (args.input / 'region_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
