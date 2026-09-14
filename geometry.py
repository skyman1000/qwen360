"""Differentiable, pixel-centred ERP geometry; x right, y up, z forward."""
import math
import torch
import torch.nn.functional as F


def yaw_resample(x, degrees):
    # Periodic linear interpolation, including at the seam. An integer shift
    # reduces to torch.roll and cannot add new information to a plain MSE.
    shift = degrees / 360 * x.shape[-1]
    lo = math.floor(shift)
    return (1 - (shift - lo)) * x.roll(lo, -1) + (shift - lo) * x.roll(lo + 1, -1)


def sample_directions(erp, xyz):
    b, c, h, w = erp.shape
    xyz = F.normalize(xyz.float(), dim=-1)
    lon = torch.atan2(xyz[..., 0], xyz[..., 2])
    lat = torch.asin(xyz[..., 1].clamp(-1, 1))
    # One-column halo makes bilinear interpolation periodic in longitude.
    u = (lon / (2 * math.pi) + .5) * w - .5 + 1
    v = (.5 - lat / math.pi) * h - .5
    grid = torch.stack((2 * (u + .5) / (w + 2) - 1, 2 * (v + .5) / h - 1), -1)
    padded = torch.cat((erp[..., -1:], erp, erp[..., :1]), -1)
    return F.grid_sample(padded.float(), grid[None].expand(b, -1, -1, -1),
                         align_corners=False, padding_mode="border")


def cubemap(erp, face_size=None):
    size = face_size or max(1, erp.shape[-2] // 2)
    r = (torch.arange(size, device=erp.device).float() + .5) / size * 2 - 1
    v, u = torch.meshgrid(r, r, indexing="ij")
    one = torch.ones_like(u)
    # F, R, B, L, U, D; row increases downwards.
    faces = [(u, -v, one), (one, -v, -u), (-u, -v, -one),
             (-one, -v, u), (u, one, v), (u, -one, -v)]
    return torch.stack([sample_directions(erp, torch.stack(face, -1)) for face in faces], 1)


def perspective(erp, yaw, pitch, fov=90., size=512):
    if not 0 < fov < 180:
        raise ValueError("FOV must lie strictly between 0 and 180")
    r = ((torch.arange(size, device=erp.device).float() + .5) / size * 2 - 1)
    v, u = torch.meshgrid(r * math.tan(math.radians(fov) / 2), r * math.tan(math.radians(fov) / 2), indexing="ij")
    yaw, pitch = math.radians(yaw), math.radians(pitch)
    x, y, z = u, -v, torch.ones_like(u)
    y, z = math.cos(pitch) * y + math.sin(pitch) * z, -math.sin(pitch) * y + math.cos(pitch) * z
    x, z = math.cos(yaw) * x + math.sin(yaw) * z, -math.sin(yaw) * x + math.cos(yaw) * z
    return sample_directions(erp, torch.stack((x, y, z), -1))
