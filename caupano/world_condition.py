"""Small vector Object/Relation/Layout encoders and linear world-token adapter."""
import math
import numpy as np
import torch
from torch import nn

PREDICATES=['supported_by','inside_room','near','above','below','left_of','right_of','front_of','behind']
ROOM_CODES='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'


class WorldCondition(nn.Module):
    def __init__(self, output_dim, width=256):
        super().__init__()
        self.width=width
        self.category=nn.Embedding(42,width)
        self.existence=nn.Embedding(2,width)
        self.object_encoder=nn.Sequential(nn.Linear(13,width),nn.SiLU(),nn.Linear(width,width))
        self.predicate=nn.Embedding(len(PREDICATES),width)
        self.frame=nn.Embedding(2,width)
        self.special_target=nn.Embedding(2,width)
        self.relation_encoder=nn.Sequential(nn.Linear(2*width+4,width),nn.SiLU(),nn.Linear(width,width))
        self.room_type=nn.Embedding(len(ROOM_CODES)+1,width)
        self.layout_encoder=nn.Sequential(nn.Linear(58,width),nn.SiLU(),nn.Linear(width,4*width))
        self.world_adapter=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,output_dim))

    def forward(self,state):
        device=self.category.weight.device
        def tensor(x,dtype=torch.float32):
            return torch.tensor(x,dtype=dtype,device=device)
        objects=state['objects']
        feats=[]
        for obj in objects:
            pos=obj['angular_position']; lon,lat=pos['longitude'],pos['latitude']
            yaw_valid=bool(obj['yaw_semantic_valid'] and obj['yaw_world'] is not None)
            yaw=obj['yaw_world'] if yaw_valid else 0.
            vis_valid=bool(obj['visibility_valid'])
            feats.append([math.sin(lon),math.cos(lon),math.sin(lat),math.cos(lat),math.log(max(pos['distance_m'],1e-4)),
                          *np.log(np.maximum(obj['size'],1e-4)).tolist(),
                          math.sin(yaw) if yaw_valid else 0.,math.cos(yaw) if yaw_valid else 0.,
                          obj['visibility_ratio'] if vis_valid else 0.,float(yaw_valid),float(vis_valid)])
        tokens=self.category(tensor([o['category_id'] for o in objects],torch.long))
        tokens=tokens+self.existence(tensor([int(o['exists']) for o in objects],torch.long))
        tokens=tokens+self.object_encoder(tensor(feats).reshape(-1,13))
        index={o['id']:i for i,o in enumerate(objects)}  # IDs only index edges, never embed scene IDs.
        relation_tokens=[]
        for rel in state['relations']:
            si=index[rel['subject']]
            subject=tokens[si]
            if isinstance(rel['object'],int):
                ti=index[rel['object']]; target=tokens[ti]
                delta=np.array(objects[ti]['center_camera'])-objects[si]['center_camera']
            else:
                target=self.special_target.weight[0 if rel['object'].startswith('floor:') else 1]
                delta=np.zeros(3)
            features=torch.cat([subject,target,tensor([*delta,float(rel['confidence'])])])
            rid=PREDICATES.index(rel['predicate'])
            frame=0 if rel['reference_frame']=='world' else 1
            relation_tokens.append(self.relation_encoder(features)+self.predicate.weight[rid]+self.frame.weight[frame])
        extent=np.array(state['room']['extent']); origin=np.array(state['camera']['position_world'])
        xyz=(extent[:3]+extent[3:])/2-origin
        layout=state['layout']
        heights=[layout[k]-origin[2] if layout[k] is not None else 0. for k in ['floor_z','ceiling_z']]
        masks=[float(layout[k] is not None) for k in ['floor_z','ceiling_z']]
        polygon=layout['polygon_xy'] or []
        points=np.zeros((16,2)); valid=np.zeros(16)
        if polygon:
            # Keep source polygon order, up to 16 evenly spaced source vertices.
            n=min(16,len(polygon)); ids=np.floor(np.linspace(0,len(polygon),n,endpoint=False)).astype(int)
            points[:n]=np.array(polygon)[ids]-origin[:2]; valid[:n]=1
        lf=[*xyz,*np.log(np.maximum(extent[3:]-extent[:3],1e-4)),*heights,*masks,*points.ravel(),*valid]
        code=state['room']['type']
        room_id=ROOM_CODES.index(code) if code in ROOM_CODES else len(ROOM_CODES)
        lt=self.layout_encoder(tensor(lf)).reshape(4,self.width)+self.room_type.weight[room_id]
        parts=[tokens,lt]
        if relation_tokens:
            parts.append(torch.stack(relation_tokens))
        return self.world_adapter(torch.cat(parts))[None]
