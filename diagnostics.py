"""No learned models: seam statistics plus cubemap/yaw/perspective inspection assets.

These diagnostics do not certify 360-degree coverage or causal correctness.
"""
import argparse
import csv
import random
from pathlib import Path

from .common import (atomic_json, check_ids, read_jsonl, resolve_image,
                     sample_seed, sha256, write_jsonl)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="Completed generated.jsonl or reference.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--face-size", type=int, default=512)
    p.add_argument("--preview-count", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--height", type=int, default=1024)
    args = p.parse_args()
    if min(args.height, args.face_size) <= 0 or args.preview_count < 0:
        raise ValueError("Invalid image or preview dimensions")
    import numpy as np
    import torch
    from PIL import Image
    from .geometry import cubemap, perspective

    rows = read_jsonl(args.manifest)
    check_ids(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    results, manual = [], []
    for i, row in enumerate(rows):
        path = resolve_image(args.manifest, row["image"])
        with Image.open(path) as source:
            if source.size != (2 * args.height, args.height):
                raise ValueError(f"Unexpected dimensions (no automatic resize): {path}")
            rgb = np.array(source.convert("RGB")).astype(np.float32) / 255
        horizontal = np.abs(rgb[:, 1:] - rgb[:, :-1]).mean(axis=(1, 2))
        seam = np.abs(rgb[:, 0] - rgb[:, -1]).mean(axis=1)
        latitude_weight = np.cos(((np.arange(args.height) + .5) / args.height - .5) * np.pi)
        w_seam = float(np.average(seam, weights=latitude_weight))
        w_horizontal = float(np.average(horizontal, weights=latitude_weight))
        result = dict(id=row["id"], image_sha256=sha256(path), seam_mae=float(seam.mean()),
                      horizontal_gradient_mae=float(horizontal.mean()),
                      area_weighted_seam_mae=w_seam, area_weighted_horizontal_mae=w_horizontal,
                      seam_to_interior_ratio=w_seam / w_horizontal if w_horizontal > 1e-8 else None)
        results.append(result)
        if i < args.preview_count:
            folder = args.output / "previews" / row["id"]
            folder.mkdir(parents=True)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None]

            def save_tensor(x, dest):
                array = (x.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype("uint8")
                Image.fromarray(array).save(dest)

            faces = cubemap(tensor, args.face_size)[0]
            for name, face in zip(("F", "R", "B", "L", "U", "D"), faces):
                save_tensor(face, folder / f"cube_{name}.png")
            sheet = torch.cat((torch.cat(tuple(faces[:3]), -1), torch.cat(tuple(faces[3:]), -1)), -2)
            save_tensor(sheet, folder / "cube_sheet.png")
            half = min(128, args.height // 2)
            seam_band = torch.cat((tensor[0, :, :, -half:], tensor[0, :, :, :half]), -1)
            save_tensor(seam_band, folder / "seam_centered.png")
            save_tensor(tensor[0].roll(args.height, -1), folder / "yaw180_preview.png")
            rng = random.Random(sample_seed(args.seed, row["id"], "per-id"))
            views = []
            # Ensure the seam and poles are inspected, then sample extra views.
            angles = [(180., 0.), (0., 70.), (0., -70.)]
            angles += [(rng.uniform(-180, 180), rng.uniform(-60, 60)) for _ in range(3)]
            for j, (yaw, pitch) in enumerate(angles):
                save_tensor(perspective(tensor, yaw, pitch, size=args.face_size)[0], folder / f"view_{j}.png")
                views.append(dict(image=f"view_{j}.png", yaw=yaw, pitch=pitch, fov=90))
            atomic_json(folder / "views.json", views)
            manual.append(dict(id=row["id"], prompt=row.get("prompt", ""),
                               full_360_scene="", seam_object_continuity="", no_duplicate_objects="",
                               coherent_floor_ceiling="", cubemap_geometry="", text_alignment="", notes=""))
    write_jsonl(args.output / "per_image.jsonl", results)
    metrics = {}
    for key in results[0]:
        if key not in ("id", "image_sha256"):
            values = [r[key] for r in results if r[key] is not None]
            metrics[key] = dict(mean=float(np.mean(values)) if values else None,
                                std=float(np.std(values)) if values else None, count=len(values))
    atomic_json(args.output / "summary.json", dict(count=len(rows), manifest_sha256=sha256(args.manifest),
                height=args.height, width=2 * args.height, face_size=args.face_size, seed=args.seed, metrics=metrics,
                limitations=["Low seam error can result from blur or a flat image.",
                              "Projection/yaw previews are deterministic transforms, not model consistency scores.",
                              "Full 360 coverage, polar geometry, identity and semantics require inspection.",
                              "No causal ability or depth/normal consistency is measured here."]))
    if manual:
        with (args.output / "manual_review.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manual[0]))
            writer.writeheader()
            writer.writerows(manual)
    print(f"Diagnostics: {args.output}; review images and manual_review.csv before deciding on adaptation")


if __name__ == "__main__":
    main()
