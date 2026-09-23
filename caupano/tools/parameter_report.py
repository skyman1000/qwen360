"""Read saved world parameter counts, and audit optimizer ownership in training."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from safetensors import safe_open


def group_name(name):
    return '.'.join(name.split('.')[:2]) if name.startswith('branches.') else name.split('.')[0]


def training_parameter_report(base, world, optimizer):
    base_params=dict(base.named_parameters())
    world_params=dict(world.named_parameters())
    base_ids={id(p) for p in base_params.values()}
    trainable={id(p) for p in world_params.values() if p.requires_grad}
    optimized=[p for group in optimizer.param_groups for p in group['params']]
    optimizer_ids={id(p) for p in optimized}
    if any(p.requires_grad for p in base_params.values()):
        raise RuntimeError('Frozen Qwen-Pano contains trainable parameters')
    if base_ids & optimizer_ids or base_ids & {id(p) for p in world_params.values()}:
        raise RuntimeError('World optimizer/module shares parameters with frozen Qwen-Pano')
    if optimizer_ids!=trainable or len(optimized)!=len(optimizer_ids):
        raise RuntimeError('Optimizer must contain exactly the trainable world parameters, once each')
    groups=defaultdict(int)
    for name,p in world_params.items():
        if p.requires_grad:groups[group_name(name)]+=p.numel()
    return dict(base_total_parameters=sum(p.numel() for p in base_params.values()),
                base_trainable_parameters=0,
                frozen_lora_parameters=sum(p.numel() for n,p in base_params.items() if 'lora_' in n),
                world_total_parameters=sum(p.numel() for p in world_params.values()),
                world_trainable_parameters=sum(groups.values()),
                world_trainable_by_module=dict(groups),
                optimizer_parameters=sum(p.numel() for p in optimized),
                optimizer_base_overlap=0,optimizer_exact_world_match=True,
                base_frozen=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads((args.run/'training_config.json').read_text())
    groups=defaultdict(int)
    with safe_open(str(args.weights),framework='pt',device='cpu') as weights:
        for key in weights.keys():
            groups[group_name(key)]+=math.prod(weights.get_slice(key).get_shape())
    total=sum(groups.values())
    if total!=cfg['trainable_parameters']:
        raise ValueError('Saved tensor count differs from recorded trainable parameter count')
    print(json.dumps(dict(saved_world_parameters=total,by_module=dict(groups),
                          matches_training_config=True,
                          base_frozen_recorded=cfg['base_frozen'],
                          scope='Reads saved world tensor shapes only; runtime optimizer ownership is audited during future training.'),indent=2))


if __name__=='__main__':main()
