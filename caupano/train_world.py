"""Single-GPU small-fit experiment: frozen Qwen-Pano, train world conditioning only."""
import argparse
import gc
import json
import random
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import AutoencoderKLQwenImage, QwenImagePipeline, QwenImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from qwen_pano.common import text_hash, sha256, require_versions
from qwen_pano.data import ERPTrainingDataset
from qwen_pano.circular import install_circular_padding
from qwen_pano.losses import panorama_loss
from .world_condition import WorldCondition
from .gated_world import GatedWorldCondition
from .tools.parameter_report import training_parameter_report


def read(p):
    return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--text-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--fixed-flow-probe',action='store_true',
                   help='Diagnostic: fixed per-image noise, middle timestep, flow-only objective')
    p.add_argument('--init-world',type=Path,help='Load world weights only; optimizer starts fresh')
    p.add_argument('--adapter-kind',choices=['append','gated'],default='append')
    p.add_argument('--residual-scale',choices=['none','hidden_rms'],default='hidden_rms')
    p.add_argument('--adam-eps',type=float,default=1e-8)
    p.add_argument('--spatial-world',action='store_true')
    p.add_argument('--balance-world-types',action='store_true')
    p.add_argument('--height',type=int,default=512)
    p.add_argument('--eval-fractions',type=float,nargs='+',default=[.5])
    args=p.parse_args()
    if args.spatial_world and args.adapter_kind!='gated':
        raise ValueError('--spatial-world requires --adapter-kind gated')
    if args.balance_world_types and (args.adapter_kind!='gated' or args.spatial_world):
        raise ValueError('--balance-world-types requires gated attention without --spatial-world')
    require_versions()
    random.seed(args.seed);torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Run training on an available CUDA GPU')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty output for a new small-fit run (no resume implemented)')
    protocol=read(args.experiment/'protocol.json')
    ckpt=Path(protocol['frozen_qwen_pano_checkpoint'])
    pc=read(ckpt/'pano_config.json'); tc=read(ckpt/'training_config.json')
    if sha256(ckpt/'pytorch_lora_weights.safetensors')!=protocol['lora_sha256']:
        raise ValueError('Frozen checkpoint differs from protocol')
    snapshot=pc['base_snapshot']; height=args.height
    if height<=0 or height%16 or any(not 0<f<1 for f in args.eval_fractions):
        raise ValueError('Height must be divisible by 16; eval fractions must be between 0 and 1')
    if args.adapter_kind=='gated' and (args.fixed_flow_probe or args.init_world):
        raise ValueError('First gated experiment starts fresh with stochastic timesteps')
    cache_cfg=read(args.text_cache/'cache_config.json')
    if cache_cfg['base_snapshot']!=snapshot or cache_cfg['max_sequence_length']!=pc['max_sequence_length']:
        raise ValueError('Text cache base/sequence length differs from frozen checkpoint')
    sets={name:[json.loads(s) for s in (args.experiment/(name+'.jsonl')).read_text().splitlines()] for name in ['fit','dev']}
    by_id={r['sample_id']:r for rows in sets.values() for r in rows}
    evaluation_sets=dict(sets)
    if protocol.get('comparison_fit_ids'):
        evaluation_sets['fit']=[by_id[sid] for sid in protocol['comparison_fit_ids']]
    if {r['sample_id'] for r in sets['fit']}&{r['sample_id'] for r in sets['dev']}:
        raise ValueError('Fit/dev overlap')
    if any(r['split']!='train' for rows in sets.values() for r in rows):
        raise ValueError('Small-fit/dev must come from source train only')
    if protocol.get('heldout_world_scan'):
        if {r['scan_id'] for r in sets['fit']}!=set(protocol['train_scans']):
            raise ValueError('Fit buildings differ from frozen experiment protocol')
        if protocol['heldout_world_scan'] in {r['scan_id'] for r in sets['fit']}:
            raise ValueError('Heldout world building leaked into fitting')
    states={r['sample_id']:read(r['world_state']) for rows in sets.values() for r in rows}
    text={}
    for rows in sets.values():
        for r in rows:
            t=load_file(str(args.text_cache/(text_hash(r['caption'])+'.safetensors')))
            text[r['sample_id']]=(t['embeddings'][None],t['mask'][None].bool())
    args.output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda:0')
    print(f'Encoding {len(by_id)} RGBs with frozen VAE; transformer not yet loaded.',flush=True)
    vae=AutoencoderKLQwenImage.from_pretrained(snapshot,subfolder='vae',torch_dtype=torch.float32,local_files_only=True).requires_grad_(False).eval().to(device)
    vae.enable_tiling()
    scale=2**len(vae.temperal_downsample)
    mean=torch.tensor(vae.config.latents_mean,device=device).reshape(1,-1,1,1,1)
    std=torch.tensor(vae.config.latents_std,device=device).reshape(1,-1,1,1,1)
    latents={}
    reference_ids=protocol.get('latent_reference_ids')
    encoding_ids=(reference_ids+[sid for sid in by_id if sid not in reference_ids]) if reference_ids else list(by_id)
    encoding_rows=[by_id[sid] for sid in encoding_ids]
    source=[dict(id=r['sample_id'],image=r['rgb'],caption=r['caption'],kind='panorama') for r in encoding_rows]
    dataset=ERPTrainingDataset(source,height,augment=False,profile='custom')
    for i,r in enumerate(encoding_rows):
        extra=reference_ids is not None and r['sample_id'] not in reference_ids
        # Encode the original fit8/dev4 first, preserving their VAE samples and
        # the RNG state used to initialize run003's world module. Extra images
        # get reproducible private RNG draws, without advancing that state.
        with torch.no_grad(),torch.random.fork_rng(devices=[0],enabled=extra):
            if extra:
                image_seed=int(hashlib.sha256(r['sample_id'].encode()).hexdigest()[:8],16)
                torch.manual_seed(args.seed+image_seed)
            z=vae.encode(dataset[i]['pixels'][None,:,None].to(device)).latent_dist.sample()
            latents[r['sample_id']]=((z.float()-mean)/std).to(torch.bfloat16).cpu()
    del vae,z,mean,std;gc.collect();torch.cuda.empty_cache()
    print('Loading frozen Qwen-Pano epoch005 transformer.',flush=True)
    base=QwenImageTransformer2DModel.from_pretrained(snapshot,subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True)
    base.add_adapter(LoraConfig(r=pc['rank'],lora_alpha=pc['lora_alpha'],target_modules=pc['targets']))
    loaded=set_peft_model_state_dict(base,load_file(str(ckpt/'adapter_resume.safetensors')))
    if loaded.unexpected_keys or any('lora_' in k for k in loaded.missing_keys):
        raise ValueError('Checkpoint LoRA state mismatch')
    base.requires_grad_(False).eval().to(device)
    install_circular_padding(base,pc['padding_columns'])
    base.enable_gradient_checkpointing()
    outdim=text[sets['fit'][0]['sample_id']][0].shape[-1]
    if args.adapter_kind=='gated':
        world=GatedWorldCondition(base.inner_dim,residual_scale=args.residual_scale,
                                  spatial_height=height//(scale*2) if args.spatial_world else None,
                                  balance_types=args.balance_world_types).to(device)
        world.attach(base)
    else:
        world=WorldCondition(outdim).to(device)
    if args.init_world:
        world.load_state_dict(load_file(str(args.init_world)),strict=True)
    optimizer=torch.optim.AdamW([p for p in world.parameters() if p.requires_grad],
                               lr=args.learning_rate,weight_decay=1e-5,eps=args.adam_eps)
    parameter_report=training_parameter_report(base,world,optimizer)
    (args.output/'parameter_report.json').write_text(json.dumps(parameter_report,indent=2)+'\n')
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(snapshot,subfolder='scheduler',local_files_only=True)
    sigmas=scheduler.sigmas.to(device); times=scheduler.timesteps.to(device)
    loss_args=SimpleNamespace(**{k:tc[k] for k in ['geometry_backend','perspective_weight','schedule','geometry_start_epoch','lambda_cube','lambda_yaw','lambda_seam']})
    loss_args.mask_reduction=tc.get('mask_reduction','valid')
    if args.fixed_flow_probe:
        loss_args.geometry_backend='periodic'
        loss_args.lambda_cube=loss_args.lambda_yaw=loss_args.lambda_seam=0.
    config=dict(protocol=str((args.experiment/'protocol.json').resolve()),checkpoint=str(ckpt),
                fit_ids=[r['sample_id'] for r in sets['fit']],dev_ids=[r['sample_id'] for r in sets['dev']],
                evaluation_fit_ids=[r['sample_id'] for r in evaluation_sets['fit']],
                height=height,steps=args.steps,learning_rate=args.learning_rate,seed=args.seed,
                output_dim=outdim,width=256,adapter_kind=args.adapter_kind,
                residual_scale=args.residual_scale if args.adapter_kind=='gated' else None,adam_eps=args.adam_eps,
                spatial_world=args.spatial_world,spatial_token_height=height//(scale*2) if args.spatial_world else None,
                balance_world_types=args.balance_world_types,
                conditioning='gated_image_residual' if args.adapter_kind=='gated' else 'append_projected_world_tokens_after_original_text',
                hidden_dim=base.inner_dim,block_indices=[20,30,40] if args.adapter_kind=='gated' else [],
                eval_fractions=args.eval_fractions,
                trainable_parameters=parameter_report['world_trainable_parameters'],
                base_frozen=parameter_report['base_frozen'],
                base_trainable_parameters=parameter_report['base_trainable_parameters'],
                optimizer_parameters=parameter_report['optimizer_parameters'],
                latent_policy='one_seeded_posterior_sample_per_image',independent_test_claim=False,
                latent_reference_ids=reference_ids,extra_latent_rng='isolated_per_sample' if reference_ids else None,
                loss=vars(loss_args),frozen_lora_sha256=protocol['lora_sha256'],
                fixed_flow_probe=args.fixed_flow_probe,
                init_world=str(args.init_world.resolve()) if args.init_world else None,
                init_world_sha256=sha256(args.init_world) if args.init_world else None)
    (args.output/'training_config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(config,flush=True)

    def prediction(sid,noisy,t,world_sid=None,use_world=True,world_override=None):
        embedding,mask=text[sid]
        embedding=embedding.to(device);mask=mask.to(device)
        if args.adapter_kind=='gated':
            world.current_tokens=None
        if use_world:
            condition=world(world_override if world_override is not None else states[world_sid or sid])
            if args.balance_world_types:
                # Preserve run003's BF16 feature rounding; keep count metadata
                # in FP32. The attention prior is the only method change.
                condition=torch.cat([condition[...,:world.width].to(embedding.dtype).float(),
                                     condition[...,world.width:]],-1)
            if not (args.spatial_world or args.balance_world_types):condition=condition.to(embedding.dtype)
            if args.adapter_kind=='gated':
                world.current_tokens=condition
            else:
                embedding=torch.cat([embedding,condition],dim=1)
                mask=torch.cat([mask,torch.ones(condition.shape[:2],device=device,dtype=torch.bool)],dim=1)
        packed=QwenImagePipeline._pack_latents(noisy,1,noisy.shape[1],noisy.shape[3],noisy.shape[4])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=base(hidden_states=packed,timestep=t/1000,encoder_hidden_states=embedding,
                        encoder_hidden_states_mask=mask,img_shapes=[[(1,noisy.shape[3]//2,noisy.shape[4]//2)]],return_dict=False)[0]
        return QwenImagePipeline._unpack_latents(output,height,height*2,scale)

    @torch.no_grad()
    def evaluate(step):
        world.eval(); scores={}; details=[]; max_world_delta=0.
        for split,rows in evaluation_sets.items():
            totals={k:[] for k in ['world','no_world','shuffled_world','position_scrambled','constant_world']}
            for i,row in enumerate(rows):
                sid=row['sample_id'];z=latents[sid].to(device)
                for fraction in args.eval_fractions:
                    generator=torch.Generator(device=device).manual_seed(args.seed+1000+i)
                    noise=torch.randn(z.shape,generator=generator,device=device,dtype=z.dtype)
                    index=min(int(len(times)*fraction),len(times)-1)
                    t=times[index:index+1]; sigma=sigmas[index].to(z.dtype)
                    noisy=(1-sigma)*z+sigma*noise
                    values={}
                    modes=['no_world'] if args.adapter_kind=='gated' and step==0 else ['no_world','world','shuffled_world','position_scrambled','constant_world']
                    sensitivity={}
                    for mode in modes:
                        other=rows[(i+1)%len(rows)]['sample_id'] if mode=='shuffled_world' else sid
                        if mode=='constant_world':other=sets['fit'][0]['sample_id']
                        override=None
                        if mode=='position_scrambled':
                            import copy
                            override=copy.deepcopy(states[sid]);src=states[sid]['objects']
                            for j,obj in enumerate(override['objects']):
                                donor=src[(j+1)%len(src)]
                                for key in ['angular_position','center_camera','center_world']:
                                    obj[key]=copy.deepcopy(donor[key])
                        pred=prediction(sid,noisy,t,other,mode!='no_world',override)
                        if mode=='no_world':reference=pred
                        if mode=='world':
                            max_world_delta=max(max_world_delta,float((pred.float()-reference.float()).abs().max()))
                            correct_prediction=pred
                        if mode in ('shuffled_world','position_scrambled','constant_world'):
                            sensitivity[mode]=float((pred.float()-correct_prediction.float()).square().mean())
                        values[mode]=float((pred.float()-(noise-z).float()).square().mean())
                    if len(modes)==1:
                        values={k:values['no_world'] for k in totals}
                    for k,v in values.items():totals[k].append(v)
                    details.append(dict(split=split,sample_id=sid,scan_id=row['scan_id'],
                                        evaluation_group=row.get('evaluation_group',split),scheduler_fraction=fraction,
                                        sigma=float(sigma),losses=values,prediction_mse_vs_correct_world=sensitivity))
            scores[split]={k:sum(v)/len(v) for k,v in totals.items()}
        key='flow_mse_fixed_noise_mid_timestep' if args.eval_fractions==[.5] else 'flow_mse_fixed_noise_multiple_timesteps'
        # Separate noise levels: a global mean can hide conditional benefits.
        stratified={}
        for split in sets:
            stratified[split]={}
            for fraction in args.eval_fractions:
                subset=[d['losses'] for d in details if d['split']==split and d['scheduler_fraction']==fraction]
                means={mode:sum(d[mode] for d in subset)/len(subset) for mode in subset[0]}
                stratified[split][str(fraction)]=dict(losses=means,
                    shuffled_minus_correct=means['shuffled_world']-means['world'],
                    correct_better_than_shuffled_count=sum(d['world']<d['shuffled_world'] for d in subset),
                    count=len(subset))
        group_scores={}
        for group in sorted({d['evaluation_group'] for d in details}):
            subset=[d for d in details if d['evaluation_group']==group]
            group_scores[group]=dict(sample_count=len({d['sample_id'] for d in subset}),
                losses={mode:sum(d['losses'][mode] for d in subset)/len(subset) for mode in subset[0]['losses']})
        record=dict(step=step,**{key:scores},per_sample=details,evaluation_groups=group_scores,
                    by_scheduler_fraction=stratified,fit_passes=step/len(sets['fit']),
                    max_world_vs_no_world_abs_delta=max_world_delta,
                    gates=world.gate_values() if args.adapter_kind=='gated' else None,
                    zero_gate_step0_modes_reused=args.adapter_kind=='gated' and step==0,
                    note='Diagnostic denoising errors, not image quality or independent test performance')
        with (args.output/'eval.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(dict(step=step,scores=scores,gates=record['gates']),flush=True);world.train()
        if args.adapter_kind=='gated' and step>0 and max_world_delta==0.:
            raise RuntimeError('World and no-world predictions are identical across evaluation samples. See eval.jsonl and branch diagnostics; do not extend this run blindly.')

    if args.adapter_kind=='gated':
        with torch.no_grad():
            sid=sets['fit'][0]['sample_id']; z=latents[sid].to(device)
            index=len(times)//2; t=times[index:index+1]
            reference=prediction(sid,z,t,use_world=False)
            closed=prediction(sid,z,t,use_world=True)
            error=float((reference.float()-closed.float()).abs().max())
            if error>1e-5:
                raise RuntimeError(f'Zero-gate identity failed: max error {error}')
            (args.output/'zero_gate_check.json').write_text(json.dumps(dict(max_abs_error=error))+'\n')
            del reference,closed,z
        world.current_tokens=None

    evaluate(0)
    rows=sets['fit'];order=[]
    for step in range(1,args.steps+1):
        if args.adapter_kind=='gated':
            for branch in world.branches.values():branch.record_diagnostics=step<=2
        if not order:
            order=list(range(len(rows)));random.shuffle(order)
        sid=rows[order.pop()]['sample_id'];z=latents[sid].to(device)
        if args.fixed_flow_probe:
            sample_index=next(i for i,r in enumerate(rows) if r['sample_id']==sid)
            generator=torch.Generator(device=device).manual_seed(args.seed+1000+sample_index)
            noise=torch.randn(z.shape,generator=generator,device=device,dtype=z.dtype)
            ix=torch.tensor([len(times)//2],device=device)
        else:
            noise=torch.randn_like(z)
            u=compute_density_for_timestep_sampling(weighting_scheme=tc['weighting_scheme'],batch_size=1,
                logit_mean=tc['logit_mean'],logit_std=tc['logit_std'],mode_scale=tc['mode_scale'])
            ix=(u*scheduler.config.num_train_timesteps).long().clamp_max(len(times)-1).to(device)
        t=times[ix];sigma=sigmas[ix].reshape(1,1,1,1,1).to(z.dtype)
        noisy=(1-sigma)*z+sigma*noise
        pred=prediction(sid,noisy,t)
        weight=(torch.ones_like(sigma) if args.fixed_flow_probe else
                compute_loss_weighting_for_sd3(weighting_scheme=tc['weighting_scheme'],sigmas=sigma))
        mask=torch.ones((1,1,z.shape[-2],z.shape[-1]),device=device)
        loss,parts=panorama_loss(pred,noise-z,mask,torch.ones(1,device=device,dtype=torch.bool),weight,
            max(0,loss_args.geometry_start_epoch),loss_args,noise=noise,clean=z)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite small-fit loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(world.parameters(),1.)
        if not torch.isfinite(grad):raise FloatingPointError('Nonfinite world gradient')
        if step in ([1,2] if args.adapter_kind=='gated' else [1]):
            groups={name:sum(float(p.grad.detach().abs().sum()) for n,p in world.named_parameters()
                            if n.startswith(name) and p.grad is not None)
                    for name in ['object_encoder','relation_encoder','layout_encoder','world_adapter']}
            gate_gradients=({k:float(v.gate.grad.abs()) if v.gate.grad is not None else 0.
                            for k,v in world.branches.items()} if args.adapter_kind=='gated' else {})
            gate_only_step=args.adapter_kind=='gated' and step==1
            routing_failed=(not any(v>0 for v in gate_gradients.values()) if gate_only_step
                            else any(v==0 for v in groups.values()))
            if routing_failed or any(p.grad is not None for p in base.parameters()):
                raise RuntimeError(f'Gradient routing failed: encoders={groups}, gates={gate_gradients}')
            diagnostics={k:v.diagnostics for k,v in world.branches.items()} if args.adapter_kind=='gated' else {}
            (args.output/f'gradient_check_step{step}.json').write_text(json.dumps(dict(world_gradients=groups,gate_gradients=gate_gradients,base_frozen=True,branches=diagnostics),indent=2)+'\n')
        optimizer.step()
        with (args.output/'train.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=step,sample_id=sid,loss=float(loss.detach()),grad_norm=float(grad),
                                   loss_components={k:float(v) for k,v in parts.items()},
                                   gates=world.gate_values() if args.adapter_kind=='gated' else None))+'\n')
        print(f'step {step}/{args.steps} loss={loss.item():.6f}',flush=True)
        if step%args.eval_every==0 or step==args.steps:
            save_file({k:v.detach().cpu().contiguous() for k,v in world.state_dict().items()},str(args.output/f'world-step{step:05d}.safetensors'))
            evaluate(step)
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(steps=args.steps))+'\n')


if __name__=='__main__':
    main()
