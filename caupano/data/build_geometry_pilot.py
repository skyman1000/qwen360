"""Build one panorama's geometric ERP assets and integrated diagnostics on CPU."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw


def read_json(path):
    return json.loads(Path(path).read_text())


def image_from_zip(ref, rgb=False):
    with ZipFile(ref['archive']) as archive:
        with Image.open(io.BytesIO(archive.read(ref['member']))) as image:
            return np.array(image.convert('RGB') if rgb else image)


def make_rays(height):
    width = 2 * height
    lon = 2*np.pi*((np.arange(width)+.5)/width-.5)
    lat = np.pi*(.5-(np.arange(height)+.5)/height)
    lon, lat = np.meshgrid(lon, lat)
    return np.stack([np.cos(lat)*np.sin(lon), np.sin(lat), np.cos(lat)*np.cos(lon)], -1)


def camera_points(obs, shape):
    h, w = shape
    k = np.array(obs['undistorted']['intrinsics'])
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    # Stored images are read top-to-bottom; conf has bottom-left origin.
    directions = np.stack([(x-k[0,2])/k[0,0], ((h-1-y)-k[1,2])/k[1,1], -np.ones_like(x)], -1)
    return directions


def depth_stats(a, b, valid):
    if not np.any(valid):
        return dict(count=0, median_abs_m=None, p90_abs_m=None, median_relative=None)
    delta = np.abs(a[valid]-b[valid])
    return dict(count=int(valid.sum()), median_abs_m=float(np.median(delta)),
                p90_abs_m=float(np.percentile(delta,90)),
                median_relative=float(np.median(delta/np.maximum(b[valid],1e-6))))


def depth_preview(depth, valid, max_m):
    color = cv2.applyColorMap((np.clip(depth/max_m,0,1)*255).astype('uint8'), cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
    color[~valid] = 0
    return color


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    cv2.setNumThreads(args.threads)
    if args.height < 16:
        raise ValueError('ERP height must be >= 16')
    index_report = read_json(args.index/'index_validation.json')
    if not index_report['ready_for_erp_alignment']:
        raise ValueError('Index validation must permit ERP alignment')
    alignment_path = args.alignment/'alignment_report.json'
    alignment = read_json(alignment_path)
    if alignment['status'] != 'CANDIDATE_REQUIRES_VISUAL_REVIEW':
        raise ValueError('A supported image-based orientation candidate is required')
    samples = [json.loads(line) for line in (args.index/'panorama_index.jsonl').read_text().splitlines()]
    sample = next(r for r in samples if r['sample_id']==alignment['sample_id'])
    basis = np.array(alignment['erp_to_world_basis_candidate'])
    origin = np.array(alignment['origin_world_candidate'])
    if not np.allclose(basis.T@basis,np.eye(3),atol=1e-6):
        raise ValueError('ERP basis is not orthogonal')
    links = read_json(sample['object_links_path'])
    regions = read_json(sample['region_index_path'])
    region_dir = Path(sample['region_index_path']).parent
    object_dir = Path(sample['object_links_path']).parent
    object_files = {r['region_id']: r['array_file'] for r in links['region_files']}
    observations = {r['observation_id']: r for r in
                    map(json.loads,Path(sample['observations_path']).read_text().splitlines())}
    h, w = args.height, args.height*2
    args.output.mkdir(parents=True,exist_ok=True)
    local_rays = make_rays(h)
    world_rays = (local_rays@basis.T).astype('float32')
    print('Building CPU ray-casting scene from all available scan region meshes...',flush=True)
    scene = o3d.t.geometry.RaycastingScene(nthreads=args.threads)
    face_labels = {}
    triangle_count = 0
    for r in regions['regions']:
        with np.load(region_dir/r['array_file'],allow_pickle=False) as data:
            vertices = np.ascontiguousarray(data['vertices_world'],dtype='float32')
            triangles = np.ascontiguousarray(data['triangles'],dtype='uint32')
            gid = scene.add_triangles(o3d.core.Tensor(vertices),o3d.core.Tensor(triangles))
            semantic = data['face_mpcat40'].copy()
            semantic_valid = data['face_semantic_valid'].copy()
        with np.load(object_dir/object_files[r['region_id']],allow_pickle=False) as data:
            instance = data['face_instance_id'].copy()
            instance_valid = data['face_instance_valid'].copy()
        face_labels[gid] = (semantic, semantic_valid, instance, instance_valid)
        triangle_count += len(triangles)
        print(f'  region {r["region_id"]}: {len(triangles)} triangles',flush=True)

    def cast(origins, directions):
        rays = np.concatenate([np.broadcast_to(origins,directions.shape),directions],axis=-1).astype('float32')
        return {k:v.numpy() for k,v in scene.cast_rays(o3d.core.Tensor(rays),nthreads=args.threads).items()}

    hit = cast(origin,world_rays)
    mesh_valid = np.isfinite(hit['t_hit']) & (hit['t_hit']>0)
    mesh_depth = np.where(mesh_valid,hit['t_hit'],0).astype('float32')
    semantic = np.zeros((h,w),dtype='int16')
    semantic_valid = np.zeros((h,w),dtype=bool)
    instance = np.zeros((h,w),dtype='int32')
    instance_valid = np.zeros((h,w),dtype=bool)
    for gid, (sem, semvalid, inst, instvalid) in face_labels.items():
        mask = mesh_valid & (hit['geometry_ids']==gid)
        ids = hit['primitive_ids'][mask].astype('int64')
        semantic[mask], semantic_valid[mask] = sem[ids], semvalid[ids]
        instance[mask], instance_valid[mask] = inst[ids], instvalid[ids]
    normal = hit['primitive_normals'].astype('float64')@basis
    normal_length = np.linalg.norm(normal,axis=-1)
    normal_valid = mesh_valid & (normal_length>1e-8)
    normal[normal_valid] /= normal_length[normal_valid,None]
    normal[~normal_valid] = 0
    normal[np.sum(normal*local_rays,axis=-1)>0] *= -1
    normal = normal.astype('float32')
    del hit

    # Sensor point-cloud splatting with nearest radial-distance z-buffer.
    sensor = np.full(h*w,np.inf,dtype='float32')
    sensor_rgb = np.zeros((h*w,3),dtype='uint8')
    source_checks = []
    for oi, oid in enumerate(sample['observation_ids']):
        obs = observations[oid]
        depth = image_from_zip(sample['depth_source'][oi]).astype('float64')/4000.0
        color = image_from_zip(sample['perspective_rgb_source'][oi],rgb=True)
        if color.shape[:2]!=depth.shape:
            raise ValueError(f'{oid}: source color/depth dimensions differ')
        directions = camera_points(obs,depth.shape)
        pose = np.array(obs['undistorted']['camera_to_world'])
        valid = depth>0
        points = (directions[valid]*depth[valid,None])@pose[:3,:3].T+pose[:3,3]
        local = (points-origin)@basis
        rho = np.linalg.norm(local,axis=-1)
        nonzero = rho>1e-6
        local, rho = local[nonzero],rho[nonzero]
        lon = np.arctan2(local[:,0],local[:,2])
        lat = np.arcsin(np.clip(local[:,1]/rho,-1,1))
        u = np.floor(w*(lon/(2*np.pi)+.5)).astype('int64')%w
        v = np.clip(np.floor(h*(.5-lat/np.pi)).astype('int64'),0,h-1)
        pixels = v*w+u
        # Sort by pixel then distance, taking one nearest source sample per bin.
        order = np.lexsort((rho,pixels))
        _, first = np.unique(pixels[order],return_index=True)
        chosen = order[first]
        bins = pixels[chosen]
        closer = rho[chosen]<sensor[bins]
        chosen,bins = chosen[closer],bins[closer]
        sensor[bins] = rho[chosen]
        sensor_rgb[bins] = color[valid][nonzero][chosen]
        # Independent camera-frame check: mesh distance at raw camera pixels
        # versus source axial depth converted to radial distance. No ERP basis.
        small = directions[::16,::16]
        length = np.linalg.norm(small,axis=-1)
        measured = depth[::16,::16]*length
        world = (small/length[...,None])@pose[:3,:3].T
        raw_hit = cast(pose[:3,3],world)
        predicted = raw_hit['t_hit']
        common = (measured>0)&np.isfinite(predicted)&(predicted>0)
        source_checks.append(dict(observation_id=oid,**depth_stats(measured,predicted,common)))
        print(f'  sensor {oi+1}/{len(sample["observation_ids"])} projected',flush=True)
    sensor_valid = np.isfinite(sensor).reshape(h,w)
    sensor = np.where(np.isfinite(sensor),sensor,0).reshape(h,w)
    sensor_rgb = sensor_rgb.reshape(h,w,3)
    # Final pilot RGB uses pixel centres, leaving the legacy RGB untouched.
    legacy = np.array(Image.open(args.alignment/'rgb_erp.png').convert('RGB'))
    lh,lw = legacy.shape[:2]
    xx,yy = np.meshgrid((np.arange(w)+.5)*(lw-1)/w,(np.arange(h)+.5)*(lh-1)/h)
    rgb = cv2.remap(legacy,xx.astype('float32'),yy.astype('float32'),cv2.INTER_LINEAR)
    Image.fromarray(rgb).save(args.output/'rgb_erp.png')
    Image.fromarray(sensor_rgb).save(args.output/'rgb_sensor_projection.png')
    # Dense coarse labels from pinned semantic mesh, not inferred room polygons.
    layout = np.zeros((h,w),dtype='uint8')
    for mpcat, label in [(2,1),(1,2),(17,3),(4,4),(9,5)]:
        layout[(semantic==mpcat)&semantic_valid] = label
    layout_valid = semantic_valid.copy()  # zero is known non-layout only where valid
    arrays = dict(depth_sensor_erp=sensor,depth_sensor_valid=sensor_valid,
                  depth_mesh_erp=mesh_depth,depth_mesh_valid=mesh_valid,
                  normal_erp=normal,normal_valid=normal_valid,
                  semantic_erp=semantic,semantic_valid=semantic_valid,
                  instance_erp=instance,instance_valid=instance_valid,
                  layout_erp=layout,layout_valid=layout_valid)
    for name,array in arrays.items():
        np.save(args.output/(name+'.npy'),array,allow_pickle=False)
    weights = np.broadcast_to((np.sin(np.pi*(.5-np.arange(h)/h))-
                               np.sin(np.pi*(.5-(np.arange(h)+1)/h)))[:,None]*(2*np.pi/w),(h,w))
    objects = []
    for obj in links['objects']:
        mask = instance_valid & (instance==obj['instance_id'])
        objects.append(dict(house_object_id=obj['house_object_id'],instance_id=obj['instance_id'],
                            category=obj['category'],semantic_valid=obj['semantic_valid'],
                            visible=bool(mask.any()),visible_pixel_count=int(mask.sum()),
                            visible_solid_angle_sr=float(weights[mask].sum()),
                            projected_pixel_count=None,visibility_ratio=None,
                            visibility_source='candidate_pose_mesh_raycast',
                            mask_reference=dict(path='instance_erp.npy',value=obj['instance_id'])))
    (args.output/'visibility.json').write_text(json.dumps(dict(objects=objects,
        note='Projected object area/visibility ratio pending; zero hit count is not proof of object absence.'),indent=2)+'\n')
    (args.output/'layout.json').write_text(json.dumps(dict(
        labels={'0':'non_layout_or_invalid_use_mask','1':'floor','2':'wall','3':'ceiling','4':'door','5':'window'},
        source='pinned_mpcat40_semantic_mesh',room_geometry=None),indent=2)+'\n')
    common = sensor_valid&mesh_valid
    statistics = depth_stats(sensor,mesh_depth,common)
    checks = dict(nonempty_sensor=bool(sensor_valid.any()),nonempty_mesh=bool(mesh_valid.any()),
                  overlapping_depth=bool(common.any()),
                  instance_ids_known=bool(set(np.unique(instance[instance_valid])).issubset(
                      {o['instance_id'] for o in links['objects']})),
                  normals_unit=bool(np.allclose(np.linalg.norm(normal[normal_valid],axis=-1),1,atol=1e-5)))
    # These are screening flags only, never auto-promote a candidate to GT.
    review_flags = []
    if statistics['count'] and statistics['median_relative']>.10:
        review_flags.append('sensor_mesh_median_relative_above_0.10')
    if sensor_valid.mean()<.25:
        review_flags.append('sensor_coverage_below_0.25')
    raw_bad = [r['observation_id'] for r in source_checks if r['count']==0 or r['median_relative']>.10]
    if raw_bad:
        review_flags.append('source_camera_depth_mesh_disagreement')
    source_files = [alignment_path,args.index/'panorama_index.jsonl',Path(sample['object_links_path']),
                    Path(sample['object_validation_path']),Path(sample['region_index_path'])]
    result = dict(sample_id=sample['sample_id'],status='BUILT_FOR_REVIEW' if all(checks.values()) else 'CHECK_FAILED',
                  height=h,width=w,triangles=triangle_count,checks=checks,review_flags=review_flags,
                  sensor_coverage=float(sensor_valid.mean()),mesh_coverage=float(mesh_valid.mean()),
                  semantic_coverage=float(semantic_valid.mean()),instance_coverage=float(instance_valid.mean()),
                  visible_linked_objects=sum(o['visible'] for o in objects),
                  sensor_mesh_depth=statistics,source_camera_depth_checks=source_checks,
                  source_camera_review_observation_ids=raw_bad,
                  depth_unit='meters',depth_definition='radial_distance',depth_png_units_per_meter=4000,
                  erp_sampling='pixel_centers',erp_to_world_basis_candidate=basis.tolist(),
                  origin_world_candidate=origin.tolist(),normal_source='mesh_triangles_facing_camera',
                  normal_frame='erp_x_right_y_up_z_forward',
                  room_supervision_eligible=sample['room_supervision_eligible'],
                  room_supervision_excluded_house_object_ids=sample['room_supervision_excluded_house_object_ids'],
                  source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                  mesh_scope='all_available_scan_regions',missing_mesh_regions=regions['house_regions_without_mesh'],
                  erp_alignment_validated=False,ready_for_training=False,
                  pending=['metric_alignment_review','projected_object_area_and_visibility_ratio',
                           'structured_layout_and_relations','scan_split_and_caption'],
                  software=dict(open3d=o3d.__version__,numpy=np.__version__),
                  data_reference='https://github.com/niessner/Matterport/blob/master/data_organization.md')
    # A single contact sheet for inspecting RGB/geometry correspondence.
    rng = np.random.default_rng(0)
    palette = rng.integers(32,256,size=(42,3),dtype='uint8')
    sem_image = palette[semantic]
    sem_image[~semantic_valid] = 0
    overlay = (.55*rgb+.45*sem_image).astype('uint8')
    normals_image = ((normal+1)*127.5).clip(0,255).astype('uint8')
    normals_image[~normal_valid] = 0
    limit = float(np.percentile(mesh_depth[mesh_valid],95)) if mesh_valid.any() else 10.
    tiles = [('RGB pixel centres',rgb),('Sensor RGB projection (holes black)',sensor_rgb),
             ('Sensor radial depth',depth_preview(sensor,sensor_valid,limit)),
             ('Mesh radial depth',depth_preview(mesh_depth,mesh_valid,limit)),
             ('RGB + semantic mesh overlay',overlay),('Mesh normal in ERP frame',normals_image)]
    sheet = Image.new('RGB',(1024,3*282),'white')
    draw = ImageDraw.Draw(sheet)
    for i,(label,img) in enumerate(tiles):
        x,y = (i%2)*512,(i//2)*282
        sheet.paste(Image.fromarray(img).resize((512,256)),(x,y+24))
        draw.text((x+4,y+4),label,fill='black')
    sheet.save(args.output/'geometry_overview.jpg',quality=94)
    (args.output/'geometry_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','source_camera_depth_checks')},indent=2))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
