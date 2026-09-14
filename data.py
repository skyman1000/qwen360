"""ERP training manifests, aligned augmentations and DiT360 staged mixing."""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .common import caption_text, check_ids, read_jsonl, resolve_image, text_hash


def training_rows(manifest, kind, heldout, allow_unverified=False, profile="custom", allow_overlap=False):
    rows = read_jsonl(manifest)
    # Original JSONL has no mandatory id/house field. Add stable bookkeeping IDs
    # without pretending that a hash identifies the Matterport house.
    for row in rows:
        if "id" not in row:
            if isinstance(row.get("image"), dict):
                raise ValueError("Cached Arrow rows require an explicit stable id")
            row["id"] = kind + "_" + text_hash(str(resolve_image(manifest, row["image"])))[:24]
        row["caption"] = caption_text(row.get("caption"))
    check_ids(rows)
    heldout_rows = read_jsonl(heldout)
    excluded = {r.get("scene_id") or r.get("scan_id") for r in heldout_rows}
    if None in excluded:
        raise ValueError("The held-out manifest needs scene_id/scan_id on every row for house-level exclusion")
    excluded_ids = {r["id"] for r in heldout_rows}
    excluded_views = {r[key] for r in heldout_rows for key in ("view_id", "source_view_id") if r.get(key)}
    for row in rows:
        if row["id"] in excluded_ids and not allow_overlap:
            raise ValueError(f"Held-out sample in training: {row['id']}")
        for key in ("source_view_id", "view_id"):
            if row.get(key) in excluded_views and not allow_overlap:
                raise ValueError(f"Held-out panorama UUID in training: {row[key]}. "
                                 "Polished and original images share sample identity; split the data before training.")
        scene = row.get("scene_id") or row.get("scan_id")
        if scene in excluded and not allow_overlap:
            raise ValueError(f"Held-out house in training: {scene}")
        if kind == "panorama" and not scene and not allow_unverified:
            raise ValueError("Missing house identity: supply scene_id/scan_id, or explicitly use "
                             "--allow-unverified-split for an exploratory run")
        if row.get("split", "train") not in ("train", "training"):
            raise ValueError(f"Non-training split in manifest: {row['id']}")
        row["kind"] = kind
        if isinstance(row["image"], dict):
            pointer = dict(row["image"])
            pointer["arrow_file"] = str(resolve_image(manifest, pointer["arrow_file"]))
            for key in ("batch_index", "row_index"):
                if type(pointer.get(key)) is not int or pointer[key] < 0:
                    raise ValueError("Invalid Arrow row pointer")
            stat = Path(pointer["arrow_file"]).stat()
            if stat.st_size != pointer["size_bytes"] or stat.st_mtime_ns != pointer["mtime_ns"]:
                raise ValueError("Arrow source changed after indexing; regenerate the index in a new directory")
            row["image"] = pointer
        else:
            row["image"] = str(resolve_image(manifest, row["image"]))
            if not Path(row["image"]).is_file():
                raise FileNotFoundError(row["image"])
        if kind == "perspective" and not row.get("mask"):
            raise ValueError("Perspective rows need an ERP image AND a projected valid-region mask")
        if row.get("mask"):
            row["mask"] = str(resolve_image(manifest, row["mask"]))
            if not Path(row["mask"]).is_file():
                raise FileNotFoundError(row["mask"])
    return rows


def epoch_rows(panoramas, perspectives, epoch, seed, schedule):
    if schedule == "paper-hybrid":
        return list(panoramas) + list(perspectives)
    if not perspectives or epoch < 2:
        return list(panoramas)
    count = max(1, int(len(perspectives) * .5 ** (epoch - 1))) if schedule == "dit360-mix" else len(perspectives)
    generator = torch.Generator().manual_seed(seed + epoch)
    indices = torch.randperm(len(perspectives), generator=generator)[:count].tolist()
    return list(panoramas) + [perspectives[i] for i in indices]


class ERPTrainingDataset(Dataset):
    def __init__(self, rows, height, augment=False, profile="custom"):
        self.rows, self.height, self.augment, self.profile = rows, height, augment, profile
        from .cached_data import ArrowImages
        self.arrow_images = ArrowImages()
        if profile == "custom" and augment and any(not r.get("orientation_invariant", False) for r in rows if r["kind"] == "panorama"):
            raise ValueError("Yaw/flip augmentation requires orientation_invariant=true on every caption")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        size = (2 * self.height, self.height)
        source_image = self.arrow_images.open(row["image"]) if isinstance(row["image"], dict) else Image.open(row["image"])
        with source_image as source:
            if source.width != 2 * source.height:
                raise ValueError(f"Not an ERP (do not stretch perspective images): {row['image']}")
            pixels = np.array(source.convert("RGB").resize(size, Image.Resampling.BICUBIC))
        mask_size = size if self.profile == "custom" else (size[0] // 8, size[1] // 8)
        if row.get("mask"):
            with Image.open(row["mask"]) as source:
                if source.width != 2 * source.height:
                    raise ValueError("Mask must also be a 2:1 ERP")
                mask = np.array(source.convert("L").resize(mask_size, Image.Resampling.NEAREST)) > 128
        else:
            mask = np.ones((mask_size[1], mask_size[0]), dtype=bool)
        if row["kind"] == "panorama" and not mask.all():
            raise ValueError("Panorama branch expects a fully valid image; use perspective for partial ERP")
        if not mask.any():
            raise ValueError(f"Empty supervision mask: {row['id']}")
        if self.augment and row["kind"] == "panorama":
            if torch.rand(1).item() < .5:
                pixels, mask = pixels[:, ::-1], mask[:, ::-1]
            shift = random.randint(size[0] // 3, size[0])
            pixels = np.roll(pixels, shift, 1)
            # Panorama masks are validated all-white. Custom full-resolution
            # masks can still be transformed synchronously.
            if self.profile == "custom":
                mask = np.roll(mask, shift, 1)
        return {"pixels": (torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255 - .5) / .5,
                "mask": torch.from_numpy(mask.copy()).unsqueeze(0).float(),
                "caption": row["caption"], "kind": row["kind"], "id": row["id"]}


def collate(rows):
    return {"pixels": torch.stack([r["pixels"] for r in rows]),
            "mask": torch.stack([r["mask"] for r in rows]),
            "captions": [r["caption"] for r in rows],
            "kinds": [r["kind"] for r in rows], "ids": [r["id"] for r in rows]}
