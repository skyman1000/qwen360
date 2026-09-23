"""Soft ERP attention priors, using canonical pixel-centre ray conventions."""
import math
import torch
from qwen_pano.circular import pad_columns


def token_geometry(state, device):
    # Match WorldCondition order: objects, four layout tokens, relations.
    objects=state['objects']; relations=state['relations']
    records=[]; by_id={}
    for obj in objects:
        p=obj['angular_position'];lon=p['longitude'];lat=p['latitude']
        direction=[math.cos(lat)*math.sin(lon),math.sin(lat),math.cos(lat)*math.cos(lon)]
        radius=.5*math.sqrt(sum(s*s for s in obj['size']))
        sigma=max(.35,min(1.2,math.atan2(radius,p['distance_m'])))
        by_id[obj['id']]=direction
        records.append([*direction,1/(sigma*sigma),-math.log(len(objects))])
    records.extend([[0.,0.,0.,0.,-math.log(4)]]*4)
    for rel in relations:
        # Broad localization at the relation's subject; not a projected mask.
        records.append([*by_id[rel['subject']],1.,-math.log(len(relations))])
    return torch.tensor(records,device=device,dtype=torch.float32)[None]


def attention_bias(metadata, height, sequence_length):
    width=height*2
    padded_width=sequence_length//height
    if sequence_length%height or padded_width<width or (padded_width-width)%2:
        raise ValueError('Spatial world bias requires the canonical ERP token grid')
    columns=(padded_width-width)//2
    device=metadata.device
    lon=((torch.arange(width,device=device,dtype=torch.float32)+.5)/width-.5)*(2*math.pi)
    lat=(.5-(torch.arange(height,device=device,dtype=torch.float32)+.5)/height)*math.pi
    lat,lon=torch.meshgrid(lat,lon,indexing='ij')
    rays=torch.stack([lat.cos()*lon.sin(),lat.sin(),lat.cos()*lon.cos()],-1).reshape(1,-1,3)
    if columns:rays=pad_columns(rays,height,width,columns)
    # cos(angle)-1 is periodic at the ERP seam and finite at the poles.
    cosine=(rays@metadata[...,:3].transpose(-1,-2)).clamp(-1,1)
    return (cosine-1)*metadata[...,3][:,None,:]+metadata[...,4][:,None,:]
