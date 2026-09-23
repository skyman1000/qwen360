"""Join assembled samples to source panorama splits without rebuilding geometry."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests',type=Path,nargs='+',required=True)
    p.add_argument('--split-root',type=Path,default=Path('benchmark_assets/Matterport3D_metadata/mp3d_skybox'))
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fit-count',type=int,default=8)
    p.add_argument('--dev-count',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    assignment={}
    for split in ['train','test']:
        for face_paths in np.load(args.split_root/(split+'.npy'),allow_pickle=False):
            value=face_paths[0].decode() if isinstance(face_paths[0],bytes) else str(face_paths[0])
            scan=value.split('/')[0]; uuid=Path(value).name.split('_')[0]
            sid=f'{scan}_{uuid}'
            if sid in assignment:
                raise ValueError(f'Duplicate source panorama assignment: {sid}')
            assignment[sid]=split
    rows={}
    for path in args.manifests:
        for line in path.read_text().splitlines():
            row=json.loads(line); sid=row['sample_id']
            if sid in rows:
                raise ValueError(f'Duplicate assembled sample: {sid}')
            if sid not in assignment:
                raise ValueError(f'Assembled panorama missing from source splits: {sid}')
            row.update(split=assignment[sid],split_status='source_panorama_assignment',
                       split_protocol='panfusion_mvdiffusion_local_panorama_lists',
                       ready_for_training=False)
            rows[sid]=row
    train=sorted((r for r in rows.values() if r['split']=='train'),key=lambda r:r['sample_id'])
    test=sorted((r for r in rows.values() if r['split']=='test'),key=lambda r:r['sample_id'])
    if min(args.fit_count,args.dev_count)<1 or args.fit_count+args.dev_count>len(train):
        raise ValueError('Need enough source-train samples for disjoint fit/dev subsets')
    order=np.random.default_rng(args.seed).permutation(len(train))
    fit=[dict(train[int(i)],experiment_role='small_fit') for i in order[:args.fit_count]]
    dev=[dict(train[int(i)],experiment_role='dev_from_source_train') for i in order[args.fit_count:args.fit_count+args.dev_count]]
    checkpoint=args.checkpoint.resolve()
    complete=json.loads((checkpoint/'COMPLETE.json').read_text())
    config=json.loads((checkpoint/'training_config.json').read_text())
    lora=checkpoint/'pytorch_lora_weights.safetensors'
    if not lora.is_file():
        raise FileNotFoundError(lora)
    prior_ids=set()
    for line in Path(config['panorama_manifest']).read_text().splitlines():
        row=json.loads(line)
        prior_ids.add((row['scene_id'],row['source_view_id']))
    def exposure(data):
        return sum((r['scan_id'],r['panorama_uuid']) in prior_ids for r in data)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,data in [('train',train),('test',test),('fit',fit),('dev',dev)]:
        (args.output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    summary=dict(protocol='panfusion_mvdiffusion_local_panorama_lists',
                 counts=dict(train=len(train),test=len(test),fit=len(fit),dev=len(dev)),
                 per_scan={scan:dict(Counter(r['split'] for r in rows.values() if r['scan_id']==scan))
                           for scan in sorted({r['scan_id'] for r in rows.values()})},
                 source_split_sha256={s:digest(args.split_root/(s+'.npy')) for s in ['train','test']},
                 assembled_manifest_sha256={str(p.resolve()):digest(p) for p in args.manifests},
                 frozen_qwen_pano_checkpoint=str(checkpoint),completed_epochs=complete['completed_epochs'],
                 lora_sha256=digest(lora),base_model=config['model'],seed=args.seed,
                 prior_base_training_overlap=dict(train=exposure(train),test=exposure(test),fit=exposure(fit),dev=exposure(dev)),
                 independent_test_claim=False,geometry_rebuilt=False,
                 intended_trainable_modules=['object_encoder','relation_encoder','layout_encoder','world_adapter'],
                 text_condition_preserved=True,ready_for_training=False,
                 remaining='Model-side Dataset, Encoders, Adapter and world-conditioned trainer integration')
    (args.output/'protocol.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
