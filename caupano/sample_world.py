"""Matched-seed baseline/world/position-permuted ERP generation from pilot weights."""
import argparse
import copy
import json
from pathlib import Path

import torch
from PIL import Image,ImageDraw
from safetensors.torch import load_file

from qwen_pano.common import text_hash,require_versions,sha256
from qwen_pano.pipeline import QwenPanoPipeline
from .world_condition import WorldCondition
from .gated_world import GatedWorldCondition


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--text-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--per-split',type=int,default=1)
    p.add_argument('--splits',nargs='+',choices=['fit','dev'],default=['fit','dev'])
    p.add_argument('--attention-report',action='store_true',help='Record actual first-step world attention without changing inference')
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--height',type=int,help='Override training height for a documented baseline comparison')
    p.add_argument('--true-cfg-scale',type=float,default=4.)
    p.add_argument('--prompt-prefix',default='')
    p.add_argument('--modes',nargs='+',choices=['no_world','world','position_scrambled','shuffled_world'],
                   default=['no_world','world','position_scrambled'])
    args=p.parse_args()
    require_versions()
    config=read(args.run/'training_config.json')
    ckpt=Path(config['checkpoint']);pc=read(ckpt/'pano_config.json')
    if sha256(ckpt/'pytorch_lora_weights.safetensors')!=config['frozen_lora_sha256']:
        raise ValueError('Frozen LoRA changed since training')
    if read(args.text_cache/'cache_config.json')['base_snapshot']!=pc['base_snapshot']:
        raise ValueError('Text cache differs from baseline snapshot')
    gated=config.get('adapter_kind')=='gated'
    if args.attention_report and not gated:
        raise ValueError('--attention-report requires a gated checkpoint')
    height=args.height or config['height']
    spatial_height=(config['spatial_token_height']*height//config['height'] if config.get('spatial_world') else None)
    world=(GatedWorldCondition(config['hidden_dim'],config['block_indices'],config['width'],
                               residual_scale=config.get('residual_scale','none'),spatial_height=spatial_height,
                               balance_types=config.get('balance_world_types',False)) if gated else
           WorldCondition(config['output_dim'],config['width'])).to('cuda:0').eval()
    world.load_state_dict(load_file(str(args.weights)),strict=True)
    pipe=QwenPanoPipeline.from_pretrained(pc['base_snapshot'],torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.load_lora_weights(str(ckpt),weight_name='pytorch_lora_weights.safetensors',adapter_name='pano')
    pipe.set_adapters('pano',adapter_weights=1.0)
    if gated:
        world.attach(pipe.transformer)
    pipe.enable_pano(pc.get('inference_padding_columns',pc['padding_columns']))
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload(gpu_id=0)
    args.output.mkdir(parents=True,exist_ok=True)
    height=args.height or config['height']
    records=[]; attention_records=[]
    for split in args.splits:
        rows=[json.loads(line) for line in (args.experiment/(split+'.jsonl')).read_text().splitlines()]
        for row_index,row in enumerate(rows[:args.per_split]):
            sid=row['sample_id'];state=read(row['world_state'])
            cached=load_file(str(args.text_cache/(text_hash(row['caption'])+'.safetensors')))
            text=cached['embeddings'][None].to('cuda:0');mask=cached['mask'][None].bool().to('cuda:0')
            actual_prompt=args.prompt_prefix+row['caption']
            if args.prompt_prefix:
                text,mask=pipe.encode_prompt(prompt=actual_prompt,device=torch.device('cuda:0'),
                                             max_sequence_length=pc['max_sequence_length'])
                if mask is None:
                    mask=torch.ones(text.shape[:2],device=text.device,dtype=torch.bool)
            edited=copy.deepcopy(state)
            for i,obj in enumerate(edited['objects']):
                donor=state['objects'][(i+1)%len(state['objects'])]
                for key in ['angular_position','center_camera','center_world']:
                    obj[key]=copy.deepcopy(donor[key])
            pictures=[]
            shuffled=read(rows[(row_index+1)%len(rows)]['world_state']) if 'shuffled_world' in args.modes else None
            for name,condition in [('no_world',None),('world',state),('position_scrambled',edited),('shuffled_world',shuffled)]:
                if name not in args.modes:
                    continue
                embedding=text;attention_mask=mask
                if gated:
                    world.current_tokens=None
                if condition is not None:
                    tokens=world(condition)
                    if config.get('balance_world_types'):
                        tokens=torch.cat([tokens[...,:world.width].to(text.dtype).float(),tokens[...,world.width:]],-1)
                    if not (config.get('spatial_world') or config.get('balance_world_types')):tokens=tokens.to(text.dtype)
                    if gated:
                        world.current_tokens=tokens
                        if args.attention_report:
                            for branch in world.branches.values():
                                branch.audit_counts=dict(objects=len(condition['objects']),layout=4,relations=len(condition['relations']))
                                branch.attention_audit=None
                    else:
                        embedding=torch.cat([text,tokens],dim=1)
                        attention_mask=torch.cat([mask,torch.ones(tokens.shape[:2],dtype=torch.bool,device=text.device)],dim=1)
                print(f'{split} {sid}: {name}',flush=True)
                image=pipe(prompt_embeds=embedding,prompt_embeds_mask=attention_mask,
                           negative_prompt=' ' if args.true_cfg_scale>1 else None,
                           max_sequence_length=pc['max_sequence_length'],
                           height=height,width=2*height,num_inference_steps=args.steps,
                           true_cfg_scale=args.true_cfg_scale,generator=torch.Generator(device='cuda:0').manual_seed(args.seed)).images[0]
                path=args.output/f'{split}_{sid}_{name}.png';image.save(path)
                if args.attention_report and condition is not None:
                    attention_records.append(dict(sample_id=sid,mode=name,prompt=actual_prompt,
                        branches={key:branch.attention_audit for key,branch in world.branches.items()}))
                pictures.append((name,image))
            with Image.open(row['rgb']) as image:
                pictures.insert(0,('source_RGB',image.convert('RGB')))
            sheet=Image.new('RGB',(1024,((len(pictures)+1)//2)*282),'white');draw=ImageDraw.Draw(sheet)
            for i,(name,image) in enumerate(pictures):
                x,y=(i%2)*512,(i//2)*282
                sheet.paste(image.resize((512,256)),(x,y+24));draw.text((x+4,y+4),name,fill='black')
            sheet.save(args.output/f'{split}_{sid}_comparison.jpg',quality=95)
            records.append(dict(sample_id=sid,split=split,prompt=row['caption'],generation_prompt=actual_prompt,seed=args.seed))
    metadata=dict(samples=records,world_weights=str(args.weights.resolve()),world_sha256=sha256(args.weights),
                  baseline_checkpoint=str(ckpt),steps=args.steps,true_cfg_scale=args.true_cfg_scale,height=height,
                  negative_prompt=' ' if args.true_cfg_scale>1 else None,lora_scale=1.,modes=args.modes,
                  adapter_kind=config.get('adapter_kind','append'),
                  residual_scale=config.get('residual_scale','none'),
                  spatial_world=config.get('spatial_world',False),
                  balance_world_types=config.get('balance_world_types',False),
                  fixed_flow_probe=config.get('fixed_flow_probe',False),
                  cfg_world_policy='same_world_in_positive_and_negative_branch' if gated else 'positive_text_tokens_only',
                  note='Matched-seed diagnostic. Position permutation is an inconsistent sensitivity control, not a valid physical counterfactual.')
    (args.output/'generation_report.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if args.attention_report:
        (args.output/'attention_report.json').write_text(json.dumps(dict(
            scope='First positive-text transformer call; 64 sampled real image queries per branch. Not all timesteps or heads individually.',
            samples=attention_records),indent=2)+'\n')


if __name__=='__main__':
    with torch.inference_mode():
        main()
