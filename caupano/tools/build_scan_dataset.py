"""Run existing panorama builders sequentially and collect a scan-level manifest."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--split-root',type=Path,default=Path('benchmark_assets/Matterport3D_metadata/mp3d_skybox'))
    p.add_argument('--caption-root',type=Path,default=Path('benchmark_assets/Matterport3D_stitched_captions/mp3d_skybox'))
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--sample-ids',type=Path,help='JSON list of source sample IDs to build')
    p.add_argument('--manifest-output',type=Path,help='Separate batch manifest directory; reuse existing per-panorama artifacts')
    p.add_argument('--split-policy',choices=['strict','defer-conflicts'],default='strict',
                   help='Defer conflicting source house splits without assigning training membership')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[3]
    index=args.index.resolve(); work=args.work_root.resolve()
    samples=[json.loads(line) for line in (index/'panorama_index.jsonl').read_text().splitlines()]
    scan=read(index/'index_manifest.json')['scan_id']
    if args.sample_ids:
        requested=read(args.sample_ids)
        available={r['sample_id']:r for r in samples}
        if not requested or len(requested)!=len(set(requested)) or set(requested)-available.keys():
            raise ValueError('Requested sample IDs must be unique and present in this canonical index')
        samples=[available[sid] for sid in requested]
    membership={}
    panorama_membership={}
    for path in sorted(args.split_root.glob('*.npy')):
        split={'val':'validation','train':'train','test':'test','validation':'validation'}.get(path.stem)
        if split is None:
            continue
        for row in np.load(path,allow_pickle=False):
            first=row[0].decode() if isinstance(row[0],bytes) else str(row[0])
            building=first.split('/')[0]
            membership.setdefault(building,set()).add(split)
            uuid=first.split('/')[-1].split('_')[0]
            panorama_membership.setdefault((building,uuid),set()).add(split)
    source_splits=sorted(membership.get(scan,set()))
    if len(source_splits)!=1 and args.split_policy=='strict':
        raise ValueError(f'{scan}: expected one scan-level split, got {membership.get(scan)}')
    split=source_splits[0] if len(source_splits)==1 else None
    split_status='source_scan_consistent' if split is not None else 'pending_scan_level_assignment'
    print(f'Scan {scan}: split={split}, source_splits={source_splits}, status={split_status}',flush=True)
    output=args.manifest_output.resolve() if args.manifest_output else work/'scan_dataset'/scan
    output.mkdir(parents=True,exist_ok=True)
    logs=output/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads))
    records=[]; results=[]
    for i,sample in enumerate(samples,1):
        uuid=sample['panorama_uuid']
        alignment=work/'erp_alignment_pilot'/scan/uuid
        geometry=work/'geometry_pilot'/scan/uuid
        world=work/'world_state_pilot'/scan/uuid
        stages=[
            ('alignment',alignment/'alignment_report.json',{'CANDIDATE_REQUIRES_VISUAL_REVIEW'},
             'qwen_pano.caupano.tools.erp_alignment_pilot',
             ['--index',str(index),'--panorama-uuid',uuid,'--output',str(alignment)]),
            ('geometry',geometry/'geometry_report.json',{'BUILT_FOR_REVIEW'},
             'qwen_pano.caupano.data.build_geometry_pilot',
             ['--index',str(index),'--alignment',str(alignment),'--output',str(geometry),'--height','512','--threads',str(args.threads)]),
            ('world',world/'build_report.json',{'BUILT'},
             'qwen_pano.caupano.data.world_state_builder',
             ['--index',str(index),'--geometry',str(geometry),'--output',str(world),'--threads',str(args.threads)])]
        result=dict(sample_id=sample['sample_id'],status='BUILDING',stages=[])
        for name,report_path,accepted,module,arguments in stages:
            print(f'[{i}/{len(samples)}] {uuid}: {name}',flush=True)
            reused=report_path.exists()
            if not reused:
                logfile=logs/f'{uuid}_{name}.log'
                with logfile.open('w') as handle:
                    completed=subprocess.run([sys.executable,'-m',module,*arguments],cwd=root,env=env,
                                             stdout=handle,stderr=subprocess.STDOUT)
                if completed.returncode:
                    result.update(status='FAILED',failed_stage=name,log=str(logfile))
                    break
            report=read(report_path)
            result['stages'].append(dict(stage=name,reused=reused,report=str(report_path),status=report['status']))
            if report['sample_id']!=sample['sample_id']:
                raise ValueError(f'Wrong sample in {report_path}')
            if report['status'] not in accepted or (name=='geometry' and (report['review_flags'] or report['height']!=512)):
                result.update(status='NEEDS_REVIEW',failed_stage=name,report=str(report_path))
                break
        else:
            caption_path=args.caption_root/scan/'blip3_stitched'/f'{uuid}.txt'
            if not caption_path.exists() or not caption_path.read_text().strip():
                result.update(status='MISSING_CAPTION',caption_path=str(caption_path))
            else:
                state=read(world/'world_state.json')
                # Keep original pilot files intact; split/caption join lives here.
                records.append(dict(sample_id=sample['sample_id'],scan_id=scan,panorama_uuid=uuid,split=split,
                                    split_status=split_status,source_scan_splits=source_splits,
                                    source_panorama_splits=sorted(panorama_membership.get((scan,uuid),set())),
                                    rgb=str((geometry/'rgb_erp.png').resolve()),world_state=str((world/'world_state.json').resolve()),
                                    geometry_directory=str(geometry),caption=caption_path.read_text().strip(),
                                    caption_source=str(caption_path.resolve()),caption_kind='existing_blip3_stitched',
                                    text_to_world_supervision=dict(pose_supervised=False,relation_supervised=False,
                                                                  room_supervised=False,object_supervised=[]),
                                    object_count=len(state['objects']),relation_count=len(state['relations']),
                                    build_status='assembled_with_derived_camera_pose',
                                    ready_for_training=False))
                result.update(status='ASSEMBLED')
        results.append(result)
        (output/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        summary=dict(scan_id=scan,split=split,split_status=split_status,source_scan_splits=source_splits,
                     requested_count=len(samples),processed_count=len(results),
                     assembled_count=len(records),not_assembled_count=len(results)-len(records),
                     samples=results,ready_for_training=False,
                     note='Build statuses are not per-panorama visual approval. Existing reports are reused; changed inputs require a new work-root.')
        save(output/'batch_report.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='samples'},indent=2))


if __name__=='__main__':
    main()
