"""Export prefixed fit/dev manifests and native-resolution pixel-centred RGB."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import hashlib


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expand-source-train',action='store_true',help='Keep original dev and fit prefix; add other source-train samples')
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty preparation output; do not overwrite a frozen experiment')
    protocol=json.loads((args.source/'protocol.json').read_text())
    sets={split:[json.loads(line) for line in (args.source/(split+'.jsonl')).read_text().splitlines()]
          for split in ['fit','dev']}
    original_fit_ids=[r['sample_id'] for r in sets['fit']]
    dev_ids=[r['sample_id'] for r in sets['dev']]
    if args.expand_source_train:
        train=[json.loads(line) for line in (args.source/'train.jsonl').read_text().splitlines()]
        known=set(original_fit_ids+dev_ids)
        sets['fit']+=sorted((r for r in train if r['sample_id'] not in known),key=lambda r:r['sample_id'])
    all_rows=sets['fit']+sets['dev']
    ids=[r['sample_id'] for r in all_rows]
    if len(ids)!=len(set(ids)) or any(r['split']!='train' for r in all_rows):
        raise ValueError('Require distinct fit/dev IDs from source train only')
    split_path=Path(protocol.get('source_split_root','benchmark_assets/Matterport3D_metadata/mp3d_skybox'))/'train.npy'
    if hashlib.sha256(split_path.read_bytes()).hexdigest()!=protocol['source_split_sha256']['train']:
        raise ValueError('Source panorama split changed')
    source_ids=set()
    for faces in np.load(split_path,allow_pickle=False):
        path=faces[0].decode() if isinstance(faces[0],bytes) else str(faces[0])
        source_ids.add(path.split('/')[0]+'_'+Path(path).name.split('_')[0])
    if not set(ids)<=source_ids:
        raise ValueError('A selected panorama is not in the original train split')
    args.output.mkdir(parents=True,exist_ok=True)
    rgbdir=args.output/'rgb';rgbdir.mkdir(exist_ok=True)
    h,w=1024,2048
    xx,yy=np.meshgrid((np.arange(w)+.5)*(w-1)/w,(np.arange(h)+.5)*(h-1)/h)
    asset_hashes={}
    for split,rows in sets.items():
        for row in rows:
            geo=json.loads((Path(row['geometry_directory'])/'geometry_report.json').read_text())
            state_path=Path(row['world_state']);state=json.loads(state_path.read_text())
            if state['sample_id']!=row['sample_id'] or geo['sample_id']!=row['sample_id'] or geo['review_flags']:
                raise ValueError(f"Source pairing or geometry flags need review: {row['sample_id']}")
            origin=np.array(state['camera']['position_world'])
            basis=np.array(state['camera']['camera_to_world'])[:3,:3]
            index={o['id'] for o in state['objects']}
            for obj in state['objects']:
                camera=(np.array(obj['center_world'])-origin)@basis
                if not np.allclose(camera,obj['center_camera'],atol=1e-5,rtol=0):
                    raise ValueError(f"Inconsistent camera coordinates: {row['sample_id']}")
            for rel in state['relations']:
                if rel['subject'] not in index or (isinstance(rel['object'],int) and rel['object'] not in index):
                    raise ValueError(f"Unresolved relation endpoint: {row['sample_id']}")
            asset_hashes[row['sample_id']]=dict(world_state=hashlib.sha256(state_path.read_bytes()).hexdigest())
            alignment=next(Path(p) for p in geo['source_sha256'] if p.endswith('/alignment_report.json'))
            legacy=alignment.parent/'rgb_erp.png'
            with Image.open(legacy) as img:
                if img.size!=(w,h):raise ValueError(f'Expected native 2048x1024 skybox ERP: {legacy}')
                rgb=np.array(img.convert('RGB'))
            centred=cv2.remap(rgb,xx.astype('float32'),yy.astype('float32'),cv2.INTER_LINEAR)
            dest=rgbdir/(row['sample_id']+'.png');Image.fromarray(centred).save(dest)
            row.update(rgb=str(dest.resolve()),rgb_source=str(legacy.resolve()),rgb_sampling='pixel_centres',
                       caption_original=row['caption'],caption='This is a panorama. '+row['caption'])
        (args.output/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    protocol.update(parent_protocol=str((args.source/'protocol.json').resolve()),
                    rgb_height=1024,prompt_prefix='This is a panorama. ',
                    remaining='Run gated-world small-fit',ready_for_training=False)
    protocol['counts'].update(fit=len(sets['fit']),dev=len(sets['dev']))
    if args.expand_source_train:
        protocol.update(latent_reference_ids=original_fit_ids+dev_ids,
                        comparison_fit_ids=original_fit_ids,fixed_dev_ids=dev_ids,
                        expansion='all_available_source_train_except_fixed_dev')
        protocol['prior_base_training_overlap']['fit']=(len(sets['fit']) if
            protocol['prior_base_training_overlap']['train']==protocol['counts']['train'] else None)
    (args.output/'preparation_report.json').write_text(json.dumps(dict(
        fit_count=len(sets['fit']),dev_count=len(sets['dev']),fit_ids=[r['sample_id'] for r in sets['fit']],
        dev_ids=dev_ids,scan_ids=sorted({r['scan_id'] for r in all_rows}),asset_hashes=asset_hashes,
        checks='source train membership, sample pairing, coordinate consistency, relation endpoints, native RGB dimensions',
        scope='Engineering preparation; derived ERP pose/relations are not newly visually approved',
        independent_test_claim=False),indent=2)+'\n')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(fit_count=len(sets['fit']),dev_count=len(sets['dev']),
                          output=str(args.output.resolve()),rgb_size=[w,h],
                          world_states_changed=False,independent_test_claim=False),indent=2))


if __name__=='__main__':main()
