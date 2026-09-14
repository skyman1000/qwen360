"""Export the official polished train set, or reproject perspective training data."""
import argparse
import json
import math
import re
from pathlib import Path

from .common import atomic_json, check_ids, read_jsonl, resolve_image, sample_seed, sha256, write_jsonl


def export_polished(args):
    from datasets import Image as DatasetImage, load_dataset, load_from_disk
    from PIL import Image
    import io
    if args.from_disk:
        ds = load_from_disk(args.dataset)
        if hasattr(ds, "keys"):
            ds = ds[args.split]
    else:
        ds = load_dataset(args.dataset, split=args.split, revision=args.revision,
                          cache_dir=args.cache_dir)
    ds = ds.cast_column("image", DatasetImage(decode=False))
    mapping = {}
    if args.scene_map:
        for row in read_jsonl(args.scene_map):
            idx = int(row["row_index"])
            if idx in mapping:
                raise ValueError("Duplicate row_index in scene mapping")
            mapping[idx] = row
        if set(mapping) != set(range(len(ds))):
            raise ValueError("Scene map must cover each dataset row exactly once")
        check_ids([{"id": mapping[i].get("id", f"polished_{i:06d}")} for i in range(len(ds))])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "images").mkdir()
    (args.output / "masks").mkdir()
    Image.new("L", (2048, 1024), 255).save(args.output / "masks/full_white.png")
    records = []
    for i, sample in enumerate(ds):
        entry = sample["image"]
        source_image_path = entry.get("path")
        # Official files retain the original Matterport panorama UUID even
        # though the top-level schema has only image/caption.
        source_match = re.fullmatch(r"([0-9a-f]{32})_clean_big_dc\.png",
                                   Path(source_image_path or "").name)
        source_view_id = source_match.group(1) if source_match else None
        image = Image.open(io.BytesIO(entry["bytes"])) if entry.get("bytes") else Image.open(entry["path"])
        row = mapping.get(i, {})
        sample_id = row.get("id", f"polished_{i:06d}")
        check_ids([{"id": sample_id}])
        relative = f"images/{sample_id}.png"
        with image:
            if image.width != 2 * image.height:
                raise ValueError(f"Official source row {i} is not 2:1")
            image.convert("RGB").save(args.output / relative)
        records.append({"id": sample_id, "image": relative, "caption": sample["caption"],
                        "mask": "masks/full_white.png",
                        "source": args.dataset, "source_row_index": i,
                        "source_image_path": source_image_path, "source_view_id": source_view_id,
                        "scene_id": row.get("scene_id") or row.get("scan_id"),
                        "split": row.get("split", args.split),
                        "split_verification": "user_mapping" if mapping else "unverified_house_identity",
                        "orientation_invariant": False})
        if (i + 1) % 100 == 0:
            print(f"Exported {i + 1}/{len(ds)}", flush=True)
    check_ids(records)
    write_jsonl(args.output / "train.jsonl", records)
    atomic_json(args.output / "provenance.json", dict(dataset=args.dataset, revision=args.revision,
                fingerprint=ds._fingerprint, split=args.split, count=len(records),
                scene_map_sha256=sha256(args.scene_map) if args.scene_map else None,
                note="Polished poles contain synthesized content; not ground-truth geometry."))


def project_perspective(args):
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    if args.height <= 0 or args.height % 16 or not 0 < args.fov < 160:
        raise ValueError("Invalid ERP height or horizontal FOV")
    if not 0 <= args.invalid_fill <= 1:
        raise ValueError("invalid-fill must lie in [0, 1]")
    if args.projection_mode == "paper" and (args.keep_aspect or args.fov != 90 or args.height != 1024):
        raise ValueError("Paper projection requires square crop, a 90-degree lateral cube face and 1024-high ERP")
    rows = read_jsonl(args.input)
    check_ids(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "images").mkdir()
    (args.output / "masks").mkdir()
    h, w = args.height, 2 * args.height
    lat = (.5 - (torch.arange(h) + .5) / h) * math.pi
    lon = ((torch.arange(w) + .5) / w - .5) * (2 * math.pi)
    result = []
    for row in rows:
        if not isinstance(row.get("caption"), str) or not row["caption"].strip():
            raise ValueError("Perspective source needs a caption")
        with Image.open(resolve_image(args.input, row["image"])) as im:
            if not args.keep_aspect:
                side = min(im.size)
                left, top = (im.width - side) // 2, (im.height - side) // 2
                im = im.crop((left, top, left + side, top + side))
            rgb = np.array(im.convert("RGB"))
        ih, iw = rgb.shape[:2]
        if min(ih, iw) < 4:
            raise ValueError("Perspective source image too small")
        tan_h = math.tan(math.radians(args.fov) / 2)
        yaw = args.yaw if args.yaw is not None else random.Random(sample_seed(args.seed, row["id"], "per-id")).uniform(-180, 180)
        phi, theta = torch.meshgrid(lat, lon - math.radians(yaw), indexing="ij")
        x, y, z = phi.cos() * theta.sin(), phi.sin(), phi.cos() * theta.cos()
        gx = x / z.clamp_min(1e-6) / tan_h
        gy = -y / z.clamp_min(1e-6) / (tan_h * ih / iw)
        valid = (z > 0) & (gx.abs() <= 1) & (gy.abs() <= 1)
        grid = torch.stack((gx, gy), -1)
        source = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255
        projected = F.grid_sample(source, grid[None], align_corners=False, padding_mode="border")[0]
        projected[:, ~valid] = args.invalid_fill
        image_path, mask_path = f"images/{row['id']}.png", f"masks/{row['id']}.png"
        Image.fromarray((projected.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype("uint8")).save(args.output / image_path)
        Image.fromarray(valid.numpy().astype("uint8") * 255).save(args.output / mask_path)
        result.append({**row, "image": image_path, "mask": mask_path,
                       "source_image": str(resolve_image(args.input, row["image"])),
                       "projection": dict(horizontal_fov=args.fov, yaw=yaw, pitch=0,
                                          center_crop_square=not args.keep_aspect,
                                          mode=args.projection_mode, invalid_fill=args.invalid_fill,
                                          implementation="local pixel-centred projection, not released author preprocessing"),
                       "kind": "perspective"})
    write_jsonl(args.output / "perspective.jsonl", result)
    atomic_json(args.output / "provenance.json", dict(input_sha256=sha256(args.input),
                horizontal_fov=args.fov, yaw=args.yaw, seed=args.seed,
                center_crop_square=not args.keep_aspect, height=h, width=w,
                mode=args.projection_mode, invalid_fill=args.invalid_fill,
                note="Perspective -> lateral ERP; white mask = supervised. No polar guidance."))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    export = sub.add_parser("polished")
    export.add_argument("--dataset", default="Insta360-Research/Matterport3D_polished")
    export.add_argument("--revision")
    export.add_argument("--cache-dir")
    export.add_argument("--from-disk", action="store_true")
    export.add_argument("--split", default="train")
    export.add_argument("--scene-map", type=Path, help="JSONL: row_index, id, scene_id, split; tied to dataset revision")
    export.add_argument("--output", type=Path, required=True)
    project = sub.add_parser("perspective")
    project.add_argument("--input", type=Path, required=True)
    project.add_argument("--output", type=Path, required=True)
    project.add_argument("--height", type=int, default=1024)
    project.add_argument("--fov", type=float, default=90)
    project.add_argument("--projection-mode", choices=("paper", "custom"), default="paper")
    project.add_argument("--invalid-fill", type=float, required=True,
                         help="Declare the [0,1] fill outside the projected face; not disclosed upstream")
    project.add_argument("--yaw", type=float, help="Fixed yaw; default is a reproducible random yaw per sample")
    project.add_argument("--seed", type=int, default=0)
    project.add_argument("--keep-aspect", action="store_true", help="Disable DiT360-style centre-square cropping")
    args = p.parse_args()
    (export_polished if args.command == "polished" else project_perspective)(args)


if __name__ == "__main__":
    main()
