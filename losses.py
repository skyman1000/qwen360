"""Per-sample supervision; never infer branch identity from a whole batch mask."""
import random
import torch
import torch.nn.functional as F

from .geometry import cubemap, yaw_resample


def masked_mse(prediction, target, mask, reduction="valid"):
    error = (prediction.float() - target.float()).square()
    denominator = mask.sum((-3, -2, -1)) * error.shape[1]
    if (denominator <= 0).any():
        raise ValueError("No fully valid latent cells remain after downsampling mask")
    if reduction == "full":
        return (error * mask).flatten(1).mean(1)
    return (error * mask).sum((-3, -2, -1)) / denominator


def panorama_loss(prediction, target, mask, pano, weighting, epoch, args, *, noise=None, clean=None):
    pred, target = prediction[:, :, 0].float(), target[:, :, 0].float()
    upstream = getattr(args, "geometry_backend", "periodic") == "upstream"
    if mask.shape[-2:] != pred.shape[-2:]:
        mask = (F.interpolate(mask.float(), size=pred.shape[-2:], mode="nearest-exact") if upstream else
                1 - F.adaptive_max_pool2d(1 - mask.float(), pred.shape[-2:]))
    mask = mask.float()
    weights = weighting.reshape(len(pred), -1)[:, 0].float()
    flow = masked_mse(pred, target, mask, getattr(args, "mask_reduction", "valid"))
    branch_weights = torch.where(pano, 1., args.perspective_weight)
    total = flow * weights * branch_weights
    log = {"flow": flow.mean().detach()}
    for name, subset in (("panorama", pano), ("perspective", ~pano)):
        if subset.any():
            log[f"flow_{name}"] = flow[subset].mean().detach()
    # Actual released schedules differ between train.py and mix training.
    geometry_on = epoch < 3 if args.schedule == "dit360-mix" else epoch >= args.geometry_start_epoch
    if pano.any() and (geometry_on or upstream):
        if upstream:
            from .vendor.dit360.cube_map import cube_map_from_equirectangular as cube_fn
            from .vendor.dit360.yaw_rotate import equirectangular_rotate_yaw as yaw_fn
            if noise is None or clean is None:
                raise ValueError("Upstream loss requires separate noise and clean latents, preserving original arithmetic")
            n, z = noise[pano, :, 0].float(), clean[pano, :, 0].float()
        else:
            cube_fn, yaw_fn = cubemap, yaw_resample
        aux = torch.zeros_like(flow[pano])
        if args.lambda_cube or upstream:
            cube_target = cube_fn(n) - cube_fn(z) if upstream else cube_fn(target[pano])
            cube = (cube_fn(pred[pano]) - cube_target).square().flatten(1).mean(1)
            aux = aux + args.lambda_cube * cube
            log["cube"] = cube.mean().detach()
        if args.lambda_yaw or upstream:
            angle = random.choice((60, 180, 300))
            yaw_target = yaw_fn(n, angle) - yaw_fn(z, angle) if upstream else yaw_fn(target[pano], angle)
            yaw = (yaw_fn(pred[pano], angle) - yaw_target).square().flatten(1).mean(1)
            aux = aux + args.lambda_yaw * yaw
            log["yaw_resampled"] = yaw.mean().detach()
        if args.lambda_seam:
            # Match the GT velocity boundary gradient, not first==last pixels.
            delta = (pred[pano, :, :, 0] - pred[pano, :, :, -1]) - (target[pano, :, :, 0] - target[pano, :, :, -1])
            seam = delta.square().flatten(1).mean(1)
            aux = aux + args.lambda_seam * seam
            log["seam_gradient"] = seam.mean().detach()
        if geometry_on:
            total = total.clone()
            total[pano] = total[pano] + weights[pano] * aux
    return total.mean(), log
