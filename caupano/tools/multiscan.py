"""Plan, build and export a bounded multi-building GT-world experiment."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARCHIVES = ['house_segmentations', 'region_segmentations', 'matterport_camera_intrinsics',
            'matterport_camera_poses', 'matterport_skybox_images', 'undistorted_camera_parameters',
            'undistorted_color_images', 'undistorted_depth_images', 'undistorted_normal_images']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def manifest(path, data):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in data))


def invoke(module, *args):
    subprocess.run([sys.executable, '-m', 'qwen_pano.'+module, *map(str, args)], cwd=ROOT, check=True)


def source_ids(path):
    result = defaultdict(list)
    for faces in np.load(path, allow_pickle=False):
        value = faces[0].decode() if isinstance(faces[0], bytes) else str(faces[0])
        scan = value.split('/')[0]
        result[scan].append(scan+'_'+Path(value).name.split('_')[0])
    return result


def ranked(values, seed):
    return sorted(values, key=lambda x: hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())


def partition(records, train_scans, heldout_scan, dev_per_scan, heldout_count, seed):
    """Hold out whole buildings from world fitting, and views within fit buildings."""
    grouped = defaultdict(list)
    seen = set()
    for row in records:
        if row['sample_id'] in seen or row['split'] != 'train':
            raise ValueError('Require unique source-train records')
        seen.add(row['sample_id'])
        grouped[row['scan_id']].append(row)
    if heldout_scan in train_scans or set(grouped) != set(train_scans+[heldout_scan]):
        raise ValueError('Building assignment differs from plan')
    fit_groups, same = [], []
    for scan in train_scans:
        indexed = {r['sample_id']: r for r in grouped[scan]}
        ordered = [indexed[sid] for sid in ranked(indexed, seed)]
        if len(ordered) < dev_per_scan+10:
            raise ValueError(f'{scan}: fewer than 10 fit samples after dev selection; review build report')
        same.extend(dict(r, evaluation_group='same_building') for r in ordered[:dev_per_scan])
        fit_groups.append([dict(r, experiment_role='world_fit') for r in ordered[dev_per_scan:]])
    fit = [group[i] for i in range(max(map(len,fit_groups))) for group in fit_groups if i < len(group)]
    indexed = {r['sample_id']: r for r in grouped[heldout_scan]}
    if len(indexed) < heldout_count:
        raise ValueError('Too few assembled heldout-building samples; review build report')
    heldout = [dict(indexed[sid], evaluation_group='heldout_building')
               for sid in ranked(indexed, seed)[:heldout_count]]
    # First two dev previews cover both evaluation groups.
    dev = [same[0], heldout[0], *same[1:], *heldout[1:]]
    return fit, dev


def plan(args):
    target = args.plan_dir.resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError('Use a new plan directory; the selection is frozen before building')
    split_root = args.split_root.resolve()
    train = source_ids(split_root/'train.npy')
    test = source_ids(split_root/'test.npy')
    if args.train_buildings < 2 or args.max_per_scan < 12:
        raise ValueError('Need at least two training buildings and 12 views per building')
    captions = args.caption_root.resolve()
    eligible = {}
    for scan, ids in train.items():
        available = []
        for sid in ids:
            path = captions/scan/'blip3_stitched'/(sid.split('_', 1)[1]+'.txt')
            if path.is_file() and path.read_text().strip():
                available.append(sid)
        eligible[scan] = available
    anchor = 'sT4fr6TAbpF'
    if len(eligible[anchor]) < 12:
        raise ValueError('Existing training building lacks source-train captions')
    # Moderate-size buildings keep full mesh parsing and raw downloads manageable.
    candidates = [s for s in train if s not in test and 60 <= len(train[s]) <= 180
                  and len(eligible[s]) >= min(60, args.max_per_scan)]
    chosen = ranked(candidates, args.seed)[:args.train_buildings]
    if len(chosen) != args.train_buildings:
        raise ValueError('Not enough caption-covered training buildings in the 60..180-view range')
    train_scans = [anchor]+chosen[:-1]
    heldout = chosen[-1]
    target.mkdir(parents=True)
    scans = []
    commands = ['#!/bin/bash', 'set -euo pipefail', 'cd '+shlex.quote(str(ROOT))]
    raw = args.raw_root.resolve()
    for scan in train_scans+[heldout]:
        limit = args.max_per_scan if scan != heldout else 12
        ids = ranked(eligible[scan], args.seed)[:limit]
        ids_path = target/(scan+'_ids.json')
        write(ids_path, ids)
        scan_dir = raw/'v1/scans'/scan
        missing = [name for name in ARCHIVES if not (scan_dir/(name+'.zip')).is_file()]
        if missing:
            commands.append(shlex.join(['python', str(ROOT/'download_mp_py3.py'), '-o', str(raw),
                                       '--id', scan, '--type', *missing]))
        scans.append(dict(scan_id=scan, role='heldout_building' if scan == heldout else 'fit_building',
                          source_train_count=len(train[scan]), requested_count=len(ids),
                          ids_path=str(ids_path), ids_sha256=digest(ids_path),
                          missing_archives=missing))
    result = dict(seed=args.seed, train_scans=train_scans, heldout_scan=heldout, scans=scans,
                  raw_root=str(raw), work_root=str(args.work_root.resolve()),
                  split_root=str(split_root), caption_root=str(captions),
                  source_split_sha256={s:digest(split_root/(s+'.npy')) for s in ['train','test']},
                  dev_per_scan=2, heldout_dev_count=4, independent_test_claim=False,
                  note='Bounded source-train subset, not a representative full-dataset benchmark')
    write(target/'plan.json', result)
    (target/'download_missing.sh').write_text('\n'.join(commands)+'\n')
    print(json.dumps(result, indent=2))


def load_plan(directory):
    data = read(directory/'plan.json')
    for split, expected in data['source_split_sha256'].items():
        if digest(Path(data['split_root'])/(split+'.npy')) != expected:
            raise ValueError('Source split changed since planning')
    for scan in data['scans']:
        if digest(scan['ids_path']) != scan['ids_sha256']:
            raise ValueError('Planned sample selection changed')
    return data


def build(args):
    directory = args.plan_dir.resolve()
    data = load_plan(directory)
    results = []
    for scan in data['scans']:
        sid = scan['scan_id']
        output = directory/'built'/sid
        try:
            invoke('caupano.tools.prepare_scan', '--scan-dir', Path(data['raw_root'])/'v1/scans'/sid,
                   '--work-root', data['work_root'], '--threads', args.threads,
                   '--split-policy', 'defer-conflicts', '--sample-ids', scan['ids_path'],
                   '--manifest-output', output, '--split-root', data['split_root'],
                   '--caption-root', data['caption_root'])
            report = read(output/'batch_report.json')
            results.append(dict(scan_id=sid, status='BUILT', requested=report['requested_count'],
                                assembled=report['assembled_count'], report=str(output/'batch_report.json')))
        except subprocess.CalledProcessError as exc:
            results.append(dict(scan_id=sid, status='FAILED', returncode=exc.returncode))
        write(directory/'build_summary.json', results)
    print(json.dumps(results, indent=2))
    if any(r['status']=='FAILED' for r in results):
        raise SystemExit('Some buildings failed. Fix the reported source issue, then rerun build; do not export yet.')


def export(args):
    directory = args.plan_dir.resolve()
    data = load_plan(directory)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new experiment output')
    paths = []
    for scan in data['scans']:
        folder = directory/'built'/scan['scan_id']
        report = read(folder/'batch_report.json')
        if report['processed_count'] != scan['requested_count'] or report['requested_count'] != scan['requested_count']:
            raise ValueError(f"Incomplete build: {scan['scan_id']}")
        path = folder/'samples.jsonl'
        planned = set(read(scan['ids_path']))
        if any(r['sample_id'] not in planned for r in rows(path)):
            raise ValueError('Built manifest contains an unplanned panorama')
        paths.append(path)
    # Reuse existing source assignment, checkpoint provenance and exposure accounting.
    source = directory/'source_protocol'
    invoke('caupano.tools.export_panfusion_protocol', '--manifests', *paths,
           '--split-root', data['split_root'], '--checkpoint', args.checkpoint,
           '--output', source, '--fit-count', 1, '--dev-count', 1, '--seed', data['seed'])
    if rows(source/'test.jsonl'):
        raise ValueError('Planned source-train export unexpectedly contains source-test samples')
    fit, dev = partition(rows(source/'train.jsonl'), data['train_scans'], data['heldout_scan'],
                         data['dev_per_scan'], data['heldout_dev_count'], data['seed'])
    protocol = read(source/'protocol.json')
    base_cfg = read(Path(protocol['frozen_qwen_pano_checkpoint'])/'training_config.json')
    prior = {(r['scene_id'], r['source_view_id']) for r in rows(base_cfg['panorama_manifest'])}
    for name, items in [('fit',fit), ('dev',dev)]:
        manifest(source/(name+'.jsonl'), items)
        protocol['counts'][name] = len(items)
        protocol['prior_base_training_overlap'][name] = sum((r['scan_id'],r['panorama_uuid']) in prior for r in items)
    # Two fit diagnostics per building, instead of evaluating hundreds of training views.
    comparison = [r['sample_id'] for scan in data['train_scans']
                  for r in [x for x in fit if x['scan_id']==scan][:2]]
    protocol.update(comparison_fit_ids=comparison, train_scans=data['train_scans'],
                    source_split_root=data['split_root'],
                    heldout_world_scan=data['heldout_scan'],
                    evaluation_groups=dict(Counter(r['evaluation_group'] for r in dev)),
                    plan_sha256=digest(directory/'plan.json'),
                    fit_counts_per_scan=dict(Counter(r['scan_id'] for r in fit)),
                    remaining='Multi-building GT-world diagnostic, not independent base-model evaluation')
    write(source/'protocol.json', protocol)
    invoke('caupano.tools.prepare_gated_fit', '--source', source, '--output', output)
    print(json.dumps(dict(experiment=str(output), fit=len(fit), dev=len(dev),
                          per_scan=protocol['fit_counts_per_scan'],
                          evaluation_groups=protocol['evaluation_groups']), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    default_plan = Path('qwen_pano/outputs/caupano/multiscan_v1')
    p = commands.add_parser('plan')
    p.add_argument('--plan-dir', type=Path, default=default_plan)
    p.add_argument('--train-buildings', type=int, default=5)
    p.add_argument('--max-per-scan', type=int, default=60)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--raw-root', type=Path, default=Path('benchmark_assets/Matterport3D_raw'))
    p.add_argument('--work-root', type=Path, default=Path('qwen_pano/outputs/caupano'))
    p.add_argument('--split-root', type=Path, default=Path('benchmark_assets/Matterport3D_metadata/mp3d_skybox'))
    p.add_argument('--caption-root', type=Path, default=Path('benchmark_assets/Matterport3D_stitched_captions/mp3d_skybox'))
    p.set_defaults(action=plan)
    p = commands.add_parser('build')
    p.add_argument('--plan-dir', type=Path, default=default_plan)
    p.add_argument('--threads', type=int, default=4)
    p.set_defaults(action=build)
    p = commands.add_parser('export')
    p.add_argument('--plan-dir', type=Path, default=default_plan)
    p.add_argument('--output', type=Path, default=Path('qwen_pano/outputs/caupano/experiments/gated_epoch005_multiscan'))
    p.add_argument('--checkpoint', type=Path, default=Path('qwen_pano/outputs/pano_official_full_59263/checkpoint-epoch005'))
    p.set_defaults(action=export)
    args = parser.parse_args()
    args.action(args)


if __name__ == '__main__':
    main()
