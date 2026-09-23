"""One-panorama legacy RGB conversion and held-out visual orientation diagnostic."""
import argparse
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from qwen_pano.geometry import perspective, sample_directions


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def fit(a, b):
    # ERP x-right/y-up/z-forward and source camera frames have different
    # handedness. Fit an orthogonal basis change, not necessarily det +1.
    u, _, vt = np.linalg.svd(a.T @ b)
    return (u @ vt).T


def angles(a, b, matrix):
    return np.degrees(np.arccos(np.clip(np.sum((a @ matrix.T) * b, axis=1), -1, 1)))


def view_rays(xy, size, yaw, pitch):
    x, v = ((xy + .5) / size * 2 - 1).T
    y, z = -v, np.ones_like(x)
    yaw, pitch = np.radians([yaw, pitch])
    y, z = np.cos(pitch)*y + np.sin(pitch)*z, -np.sin(pitch)*y + np.cos(pitch)*z
    x, z = np.cos(yaw)*x + np.sin(yaw)*z, -np.sin(yaw)*x + np.cos(yaw)*z
    return unit(np.stack([x, y, z], -1))


def read_image(ref):
    with ZipFile(ref['archive']) as archive:
        return np.array(Image.open(io.BytesIO(archive.read(ref['member']))).convert('RGB'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--panorama-uuid', default='03de84eb12c24a93bdfd88e46a6db25a')
    args = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    report = json.loads((args.index / 'index_validation.json').read_text())
    if not report['ready_for_erp_alignment']:
        raise ValueError('Canonical index validation must pass first')
    rows = [json.loads(s) for s in (args.index / 'panorama_index.jsonl').read_text().splitlines()]
    row = next(r for r in rows if r['panorama_uuid'] == args.panorama_uuid)
    observations = {r['observation_id']: r for r in
                    map(json.loads, Path(row['observations_path']).read_text().splitlines())}
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / 'benchmark_assets/PanFusion'))
    from external import py360convert
    faces = {key: read_image(ref) for key, ref in zip('ULFRBD', row['rgb_source'])}
    faces['R'] = np.flip(faces['R'], 1)
    faces['B'] = np.flip(faces['B'], 1)
    faces['U'] = np.rot90(np.flip(faces['U'], 0), 1)
    faces['D'] = np.rot90(faces['D'], 1)
    rgb = py360convert.c2e(py360convert.cube_dict2h(faces), 1024, 2048,
                           mode='bilinear', cube_format='horizon').astype(np.uint8)
    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(args.output / 'rgb_erp.png')
    # Preserve legacy RGB; resample endpoint lattice onto pixel centres ONLY
    # for use with qwen_pano.geometry. No training images are modified.
    h, w = rgb.shape[:2]
    xx, yy = np.meshgrid((np.arange(w)+.5)*(w-1)/w, (np.arange(h)+.5)*(h-1)/h)
    centered = cv2.remap(rgb, xx.astype('float32'), yy.astype('float32'), cv2.INTER_LINEAR)
    erp = torch.from_numpy(centered.copy()).permute(2, 0, 1)[None].float()
    sift = cv2.SIFT_create(nfeatures=1800)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    virtual = []
    size = 512
    for yaw, pitch in [(0,0), (90,0), (180,0), (-90,0), (0,90), (0,-90)]:
        view = perspective(erp, yaw, pitch, fov=90, size=size)[0].permute(1,2,0).numpy().clip(0,255).astype('uint8')
        kp, desc = sift.detectAndCompute(cv2.cvtColor(view, cv2.COLOR_RGB2GRAY), None)
        virtual.append((yaw, pitch, kp, desc))
    matches_a, matches_b, match_obs = [], [], []
    previews, camera_world_rays, summaries = [], [], []
    for oi, (oid, ref) in enumerate(zip(row['observation_ids'], row['perspective_rgb_source'])):
        obs = observations[oid]
        original = read_image(ref)
        oh, ow = original.shape[:2]
        resized = cv2.resize(original, (640, round(oh*640/ow)))
        rh, rw = resized.shape[:2]
        kp, desc = sift.detectAndCompute(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY), None)
        # conf K has bottom-left origin and conf c2w has z backwards.
        k = np.array(obs['undistorted']['intrinsics'])
        k[1,2] = oh-1-k[1,2]
        basis = np.array(obs['undistorted']['camera_to_world'])[:3,:3] @ np.diag([1,-1,-1])
        def world_rays(xy):
            pixels = (xy+.5)*np.array([ow/rw, oh/rh])-.5
            rays = np.c_[(pixels[:,0]-k[0,2])/k[0,0], (pixels[:,1]-k[1,2])/k[1,1], np.ones(len(xy))]
            return unit(rays @ basis.T)
        count = 0
        # Keep at most one (best-ratio) virtual match per source keypoint.
        best = {}
        if desc is not None:
            for yaw, pitch, vkp, vd in virtual:
                if vd is None or len(vd) < 2:
                    continue
                for m, n in matcher.knnMatch(desc, vd, k=2):
                    if m.distance < .70*n.distance:
                        ratio = m.distance / n.distance
                        if m.queryIdx not in best or ratio < best[m.queryIdx][0]:
                            best[m.queryIdx] = (ratio, yaw, pitch, vkp[m.trainIdx].pt)
            for qi, (_, yaw, pitch, point) in best.items():
                matches_a.append(view_rays(np.array([point]), size, yaw, pitch)[0])
                matches_b.append(world_rays(np.array([kp[qi].pt]))[0])
                match_obs.append(oi)
                count += 1
        previews.append(resized)
        x, y = np.meshgrid(np.arange(rw), np.arange(rh))
        camera_world_rays.append(world_rays(np.stack([x,y],-1).reshape(-1,2)).reshape(rh,rw,3))
        summaries.append(dict(observation_id=oid, match_count=count, fit_observation=obs['yaw_index']%2 == 0))
        print(f'[{oi+1}/{len(row["observation_ids"])}] {oid}: {count} matches', flush=True)
    a, b, group = np.array(matches_a), np.array(matches_b), np.array(match_obs, dtype=int)
    train_groups = np.array([s['fit_observation'] for s in summaries])
    train = train_groups[group]
    result = dict(sample_id=row['sample_id'], status='INSUFFICIENT_MATCHES',
                  observations=summaries, erp_alignment_validated=False, ready_for_training=False,
                  scope='image-based orientation candidate; translation, depth and mesh alignment not validated',
                  rgb_sampling='legacy endpoint grid; internal pixel-center resampling for geometry',
                  fit_rule='even source yaw_index; odd yaw_index held out',
                  angular_inlier_threshold_deg=2.5, converter_path=str(Path(py360convert.__file__).resolve()))
    if train.sum() >= 12:
        rng = np.random.default_rng(0)
        ids = np.flatnonzero(train)
        best = np.zeros(len(a), dtype=bool)
        for _ in range(1500):
            subset = rng.choice(ids, 4, replace=False)
            matrix = fit(a[subset], b[subset])
            keep = (angles(a,b,matrix)<2.5) & train
            if keep.sum() > best.sum():
                best = keep
        if best.sum() >= 12:
            matrix = fit(a[best], b[best])
            error = angles(a,b,matrix)
            held = (~train) & (error<2.5)
            covered = len(set(group[held].tolist()))
            strong = best.sum()>=60 and held.sum()>=30 and covered>=3
            result.update(status='CANDIDATE_REQUIRES_VISUAL_REVIEW' if strong else 'WEAK_MATCH_REQUIRES_REVIEW',
                          erp_to_world_basis_candidate=matrix.tolist(), basis_determinant=float(np.linalg.det(matrix)),
                          origin_world_candidate=row['camera_xyz'], origin_source=row['camera_xyz_source'],
                          fit_inliers=int(best.sum()), heldout_inliers=int(held.sum()),
                          heldout_observations_with_inliers=covered,
                          heldout_match_count=int((~train).sum()),
                          heldout_inlier_median_deg=float(np.median(error[held])) if held.any() else None)
            # All observation rows: source, reprojection, 50/50 overlay.
            sheet = Image.new('RGB', (960, len(previews)*286), 'white')
            draw = ImageDraw.Draw(sheet)
            for i, (original, rays, summary) in enumerate(zip(previews,camera_world_rays,summaries)):
                local = rays @ matrix
                projected = sample_directions(erp, torch.from_numpy(local).float())[0].permute(1,2,0).numpy().clip(0,255).astype('uint8')
                overlay = (original.astype(float)*.5 + projected.astype(float)*.5).astype('uint8')
                for j, img in enumerate([original, projected, overlay]):
                    sheet.paste(Image.fromarray(img).resize((320,256)), (j*320,i*286+25))
                draw.text((4,i*286+4), f'{i}: {summary["observation_id"].rsplit("_",2)[-2:]} / {"FIT" if summary["fit_observation"] else "HELD OUT"} | source / ERP / overlay', fill='black')
                summary['angular_inliers'] = int(((group==i)&(error<2.5)).sum())
            sheet.save(args.output/'alignment_contact_sheet.jpg', quality=92)
    (args.output/'alignment_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='observations'},indent=2))


if __name__ == '__main__':
    with torch.inference_mode():
        main()
