"""Build structured world state, isolated-mesh visibility and derived relations."""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from .build_geometry_pilot import make_rays, read_json


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def obb(obj):
    axes = np.array(obj['axes_world_source'],dtype=float)
    axes = np.vstack([axes,np.cross(axes[0],axes[1])])
    axes /= np.linalg.norm(axes,axis=1,keepdims=True)
    radii = np.array(obj['radii'])
    signs = np.array(list(itertools.product([-1,1],repeat=3)))
    corners = np.array(obj['center_world'])+(signs*radii)@axes
    horizontal = np.flatnonzero(np.abs(axes[:,2])<.25)
    axis = int(horizontal[np.argmax(radii[horizontal])]) if len(horizontal) else None
    yaw = float(np.arctan2(axes[axis,1],axes[axis,0])%np.pi) if axis is not None else None
    return axes,corners,yaw


def inside(poly, point):
    return len(poly)>=3 and cv2.pointPolygonTest(np.array(poly,dtype='float32'),tuple(map(float,point)),False)>=0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-visible-pixels',type=int,default=16)
    p.add_argument('--relation-neighbors',type=int,default=4)
    p.add_argument('--threads',type=int,default=4)
    args = p.parse_args()
    cv2.setNumThreads(args.threads)
    geo = read_json(args.geometry/'geometry_report.json')
    if geo['status']!='BUILT_FOR_REVIEW' or not all(geo['checks'].values()) or geo['review_flags']:
        raise ValueError('Geometry has unresolved build checks or review flags; inspect that report first')
    rows = [json.loads(line) for line in (args.index/'panorama_index.jsonl').read_text().splitlines()]
    sample = next(r for r in rows if r['sample_id']==geo['sample_id'])
    linked = read_json(sample['object_links_path'])
    house = read_json(Path(linked['house_directory'])/'house.json')
    regions = read_json(sample['region_index_path'])
    region_dir = Path(sample['region_index_path']).parent
    object_dir = Path(sample['object_links_path']).parent
    rooms = {r['id']:r for r in house['regions']}
    objects = {r['house_object_id']:r for r in linked['objects']}
    file_by_region = {r['region_id']:r['array_file'] for r in linked['region_files']}
    excluded = set(geo['room_supervision_excluded_house_object_ids'])
    basis = np.array(geo['erp_to_world_basis_candidate'])
    origin = np.array(geo['origin_world_candidate'])
    h,w = geo['height'],geo['width']
    directions = (make_rays(h)@basis.T).reshape(-1,3)
    instance = np.load(args.geometry/'instance_erp.npy',allow_pickle=False)
    instance_valid = np.load(args.geometry/'instance_valid.npy',allow_pickle=False)
    visible_counts = np.bincount(instance[instance_valid],minlength=max(o['instance_id'] for o in objects.values())+1)
    projected = {}
    print('Computing unoccluded object mesh projections for visibility ratios...',flush=True)
    for row in regions['regions']:
        with np.load(region_dir/row['array_file'],allow_pickle=False) as data:
            vertices = data['vertices_world'].copy()
            faces = data['triangles'].copy()
        with np.load(object_dir/file_by_region[row['region_id']],allow_pickle=False) as data:
            ids = data['face_instance_id'].copy()
        for iid in np.unique(ids[ids>0]):
            triangles = faces[ids==iid]
            used,inverse = np.unique(triangles,return_inverse=True)
            xyz = np.ascontiguousarray(vertices[used],dtype='float32')
            triangles = inverse.reshape(-1,3).astype('uint32')
            # Conservative mesh AABB ray filter; actual projected area comes from
            # isolated triangle intersections, not OBB area or a visibility guess.
            safe_d = np.where(np.abs(directions)>1e-12,directions,np.copysign(1e-12,directions))
            t0 = (xyz.min(0)-1e-5-origin)/safe_d
            t1 = (xyz.max(0)+1e-5-origin)/safe_d
            enter = np.minimum(t0,t1).max(1)
            leave = np.maximum(t0,t1).min(1)
            candidate = leave>=np.maximum(enter,0)
            count = 0
            if candidate.any():
                scene = o3d.t.geometry.RaycastingScene(nthreads=args.threads)
                scene.add_triangles(o3d.core.Tensor(xyz),o3d.core.Tensor(triangles))
                rays = np.c_[np.broadcast_to(origin,(int(candidate.sum()),3)),directions[candidate]].astype('float32')
                distance = scene.cast_rays(o3d.core.Tensor(rays),nthreads=args.threads)['t_hit'].numpy()
                count = int((np.isfinite(distance)&(distance>0)).sum())
                del scene
            if int(iid) in projected:
                raise ValueError('Instance spans multiple local meshes; union projection needs explicit handling')
            projected[int(iid)] = count
        print(f'  region {row["region_id"]}: projected {len(np.unique(ids[ids>0]))} objects',flush=True)
    polygons = {}
    floor_levels = {}
    for surface in house['surfaces']:
        if surface['label']!='F':
            continue
        points = [v['position_world'] for v in sorted(house['vertices'],key=lambda r:r['id'])
                  if v['surface_id']==surface['id']]
        if len(points)>=3:
            polygons.setdefault(surface['region_id'],[]).append([p[:2] for p in points])
            floor_levels.setdefault(surface['region_id'],[]).append(float(np.median(np.array(points)[:,2])))
    primary_room = rooms[sample['region_id']]
    room_polys = polygons.get(sample['region_id'],[])
    floor_z = float(np.median(floor_levels[sample['region_id']])) if sample['region_id'] in floor_levels else None
    # Ceiling from semantic mesh hits in the annotated room footprint.
    semantic = np.load(args.geometry/'semantic_erp.npy',allow_pickle=False)
    depth = np.load(args.geometry/'depth_mesh_erp.npy',allow_pickle=False)
    ceiling_pixels = np.flatnonzero((semantic==17)&(depth>0))
    ceiling_points = origin+directions[ceiling_pixels]*depth.reshape(-1)[ceiling_pixels,None]
    ceiling_zs = [float(p[2]) for p in ceiling_points if any(inside(poly,p[:2]) for poly in room_polys)]
    ceiling_z = float(np.median(ceiling_zs)) if len(ceiling_zs)>=16 else None
    result_objects, cache = [], {}
    ratio_issues = []
    for hid,obj in objects.items():
        iid = obj['instance_id']
        axes,corners,yaw = obb(obj)
        centre = np.array(obj['center_world'])
        camera_centre = (centre-origin)@basis
        distance = float(np.linalg.norm(camera_centre))
        lon = float(np.arctan2(camera_centre[0],camera_centre[2]))
        lat = float(np.arctan2(camera_centre[1],np.hypot(camera_centre[0],camera_centre[2])))
        nvis = int(visible_counts[iid])
        nproj = projected.get(iid)
        valid_ratio = nproj is not None and nproj>0 and nvis<=nproj
        if nproj is not None and nvis>nproj:
            ratio_issues.append(hid)
        ratio = nvis/nproj if valid_ratio else None
        membership = hid not in excluded and obj['region_id']>=0
        room = rooms.get(obj['region_id'])
        containment = bool(membership and room is not None and
            any(inside(poly,centre[:2]) for poly in polygons.get(obj['region_id'],[])) and
            room['bbox_world'][2]-.05<=centre[2]<=room['bbox_world'][5]+.05)
        eligible = bool(nvis>=args.min_visible_pixels and obj['semantic_valid'] and obj['has_region_instance_link'])
        record = dict(id=iid,house_object_id=hid,category=obj['category'],category_id=obj['mpcat40_id'],
                      exists=True,region_id=obj['region_id'],center_world=centre.tolist(),
                      center_camera=camera_centre.tolist(),axes_world=axes.tolist(),radii=obj['radii'],
                      size=(2*np.array(obj['radii'])).tolist(),yaw_world=yaw,
                      yaw_definition='longest_near_horizontal_OBB_axis_mod_pi_not_semantic_front',
                      yaw_semantic_valid=False,angular_position=dict(longitude=lon,latitude=lat,distance_m=distance),
                      visible=nvis>0,visible_pixel_count=nvis,projected_pixel_count=nproj,
                      visibility_ratio=ratio,visibility_valid=valid_ratio,
                      room_supervision_eligible=membership,inside_room_geometry_confirmed=containment,
                      conditioning_eligible=eligible,
                      supervision_mask=dict(category=obj['semantic_valid'],existence=True,position=True,size=True,
                                            semantic_yaw=False,visibility=valid_ratio,room=containment),
                      provenance=dict(category=dict(source='gt',method='pinned_mpcat40_mapping'),
                                      pose=dict(source='gt',method='house_OBB'),
                                      visibility=dict(source='derived',method='full_scene_vs_isolated_mesh_same_ERP_rays'),
                                      yaw=dict(source='derived',method='horizontal_OBB_axis',confidence=None)))
        result_objects.append(record)
        hull = cv2.convexHull(corners[:,:2].astype('float32'))
        cache[iid] = dict(corners=corners,hull=hull,area=float(cv2.contourArea(hull)),
                          vertical_axis_aligned=float(np.max(np.abs(axes[:,2])))>.95)
    selected = [o for o in result_objects if o['conditioning_eligible']]
    relations = []
    def relation(a,predicate,b,frame,method,confidence):
        relations.append(dict(subject=a,predicate=predicate,object=b,reference_frame=frame,
                              source='derived',method=method,confidence=confidence,
                              confidence_kind='heuristic_not_calibrated'))
    for obj in selected:
        iid = obj['id']; rid = obj['region_id']; c = cache[iid]['corners']
        if obj['inside_room_geometry_confirmed']:
            relation(iid,'inside_room',f'room:{rid}','world','center_in_source_floor_polygon_and_region_z',.8)
            floor = floor_levels.get(rid,[])
            if floor and cache[iid]['vertical_axis_aligned'] and obj['category_id'] not in (1,2,4,9,17):
                if min(abs(float(c[:,2].min())-z) for z in floor)<=.08:
                    relation(iid,'supported_by',f'floor:{rid}','world','OBB_bottom_floor_gap_le_0.08m',.7)
    # Candidate object support uses contact height, projected footprint overlap,
    # and approximately vertical OBB axes. No category-only support guesses.
    for a,b in itertools.permutations(selected,2):
        ca,cb = cache[a['id']],cache[b['id']]
        if a['category_id'] in (1,2,4,9,17) or b['category_id'] in (1,4,9,17):
            continue
        if not (ca['vertical_axis_aligned'] and cb['vertical_axis_aligned']):
            continue
        gap = float(ca['corners'][:,2].min()-cb['corners'][:,2].max())
        if abs(gap)<=.08 and a['center_world'][2]>b['center_world'][2]:
            overlap,_ = cv2.intersectConvexConvex(ca['hull'],cb['hull'])
            if ca['area']>0 and overlap/ca['area']>=.25:
                relation(a['id'],'supported_by',b['id'],'world','OBB_contact_and_footprint_overlap_ge_0.25',.6)
    by_id = {o['id']:o for o in selected}
    pair_ids = set()
    for a in selected:
        neighbours = sorted((b for b in selected if b['id']!=a['id']),
                            key=lambda b:np.linalg.norm(np.array(a['center_world'])-b['center_world']))
        pair_ids.update(tuple(sorted([a['id'],b['id']])) for b in neighbours[:args.relation_neighbors])
    extent = np.array(primary_room['bbox_world'])
    near_threshold = float(np.clip(.15*np.linalg.norm(extent[3:]-extent[:3]),.5,2.))
    for aid,bid in sorted(pair_ids):
        a,b = by_id[aid],by_id[bid]
        delta = np.array(a['center_world'])-b['center_world']
        if np.linalg.norm(delta)<near_threshold:
            relation(aid,'near',bid,'world','center_distance_below_room_scaled_threshold',.7)
            relation(bid,'near',aid,'world','center_distance_below_room_scaled_threshold',.7)
        for x,y in [(a,b),(b,a)]:
            xc,yc = cache[x['id']]['corners'],cache[y['id']]['corners']
            if xc[:,2].min()>yc[:,2].max()+.05:
                relation(x['id'],'above',y['id'],'world','separated_OBB_vertical_extents',.8)
                relation(y['id'],'below',x['id'],'world','separated_OBB_vertical_extents',.8)
        dl = a['angular_position']['longitude']-b['angular_position']['longitude']
        dl = float(np.arctan2(np.sin(dl),np.cos(dl)))
        if np.radians(15)<abs(dl)<np.radians(165):
            predicate,reverse = ('right_of','left_of') if dl>0 else ('left_of','right_of')
            relation(aid,predicate,bid,'panorama_camera','wrapped_longitude_difference',.7)
            relation(bid,reverse,aid,'panorama_camera','wrapped_longitude_difference',.7)
        dz = a['center_camera'][2]-b['center_camera'][2]
        if abs(dz)>.25:
            predicate,reverse = ('front_of','behind') if dz>0 else ('behind','front_of')
            relation(aid,predicate,bid,'panorama_camera','camera_positive_z_center_order_margin_0.25m',.6)
            relation(bid,reverse,aid,'panorama_camera','camera_positive_z_center_order_margin_0.25m',.6)
    pose = np.eye(4);pose[:3,:3]=basis;pose[:3,3]=origin
    camera = dict(position_world=origin.tolist(),camera_to_world=pose.tolist(),
                  frame='erp_x_right_y_up_z_forward',basis_determinant=float(np.linalg.det(basis)),
                  provenance=dict(source='derived',method='image_bearing_fit_plus_house_P_origin'),
                  alignment_status='pilot_geometry_consistent_not_dataset_wide_certified')
    layout = dict(floor_z=floor_z,ceiling_z=ceiling_z,
                  polygon_xy=room_polys[0] if len(room_polys)==1 else None,polygons_xy=room_polys,
                  floor_source='derived_median_source_floor_vertex_z',
                  polygon_source='gt_house_F_vertices_in_source_id_order',
                  ceiling_source='derived_median_ceiling_mesh_hits_inside_room_footprint',
                  dense_layout_path=str((args.geometry/'layout_erp.npy').resolve()))
    state = dict(schema_version=1,sample_id=sample['sample_id'],
                 room=dict(region_id=sample['region_id'],type=sample['region_type'],
                           type_encoding='source_Matterport_region_code',extent=primary_room['bbox_world'],
                           supervision_eligible=sample['room_supervision_eligible']),
                 camera=camera,layout=layout,objects=selected,relations=relations,
                 object_selection=dict(method='visible_semantic_valid_linked_objects',min_visible_pixels=args.min_visible_pixels,
                                       all_source_objects_path='objects.json',hidden_objects_are_not_absent=True),
                 assets=dict(geometry_directory=str(args.geometry.resolve()),
                             rgb=str((args.geometry/'rgb_erp.png').resolve())),
                 provenance=dict(object_geometry='gt',camera_alignment='derived',relations='derived'),
                 split=sample['split'],caption_source=sample['caption_source'],ready_for_training=False)
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'world_state.json',state)
    write_json(args.output/'objects.json',dict(objects=result_objects))
    write_json(args.output/'relations.json',dict(relations=relations))
    write_json(args.output/'camera.json',camera)
    write_json(args.output/'layout.json',layout)
    np.savez_compressed(args.output/'visibility.npz',
        instance_id=np.array([o['id'] for o in result_objects],dtype='int32'),
        visible_pixel_count=np.array([o['visible_pixel_count'] for o in result_objects],dtype='int32'),
        projected_pixel_count=np.array([o['projected_pixel_count'] if o['projected_pixel_count'] is not None else -1 for o in result_objects],dtype='int32'),
        visibility_ratio=np.array([o['visibility_ratio'] if o['visibility_ratio'] is not None else np.nan for o in result_objects],dtype='float32'),
        visibility_valid=np.array([o['visibility_valid'] for o in result_objects],dtype=bool))
    summary = dict(status='BUILT' if not ratio_issues else 'BUILT_WITH_VISIBILITY_ISSUES',sample_id=sample['sample_id'],
                   source_object_count=len(result_objects),conditioning_object_count=len(selected),
                   relation_count=len(relations),visibility_ratio_issue_house_object_ids=ratio_issues,
                   floor_z=floor_z,ceiling_z=ceiling_z,near_threshold_m=near_threshold,
                   relation_neighbors=args.relation_neighbors,ready_for_training=False,
                   remaining=['multi_panorama_dataset','scan_split_and_captions','encoder_adapter_training_integration'],
                   limitations=['OBB_yaw_is_not_semantic_front','room_type_retains_source_code',
                                'relations_are_geometry_heuristics_not_human_GT'])
    write_json(args.output/'build_report.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
