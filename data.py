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
    def __init__(self, rows, height, augment=False, profile="custom", hf_dataset=None, seed=0):
        self.rows, self.height, self.augment, self.profile = rows, height, augment, profile
        self.hf_dataset, self.seed = hf_dataset, seed
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.worker_epoch = None
        from .cached_data import ArrowImages
        self.arrow_images = ArrowImages()
        if profile == "custom" and augment and any(not r.get("orientation_invariant", False) for r in rows if r["kind"] == "panorama"):
            raise ValueError("Yaw/flip augmentation requires orientation_invariant=true on every caption")

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch.fill_(epoch)

    def __getitem__(self, index):
        # Persistent workers need the new epoch even though their dataset copy
        # is not recreated. Seed at epoch boundaries for reproducible resumes.
        worker = torch.utils.data.get_worker_info()
        epoch = int(self.epoch.item())
        if self.hf_dataset is not None and worker is not None and self.worker_epoch != epoch:
            seed = self.seed + epoch * max(1, worker.num_workers) + worker.id
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
            self.worker_epoch = epoch
        row = self.rows[index]
        size = (2 * self.height, self.height)
        if self.hf_dataset is not None:
            example = self.hf_dataset[row["source_row_index"]]
            if caption_text(example["caption"]) != row["caption"]:
                raise ValueError("Hugging Face caption differs from the text-cache index")
            source_image = example["image"]
        else:
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


def load_polished_dataset(rows, revision):
    """Use the upstream HF loader; the old index only verifies identity/cache keys."""
    from datasets import load_dataset
    dataset = load_dataset("Insta360-Research/Matterport3D_polished",
                           revision=revision, split="train", keep_in_memory=False)
    if not {"image", "caption"}.issubset(dataset.column_names):
        raise ValueError("Official polished dataset must contain image and caption")
    if len(dataset) != len(rows):
        raise ValueError("Official full dataset and text-cache index have different row counts")
    # In offline mode HF may fall back to the latest cached revision. Verify
    # that it really selected the shards used by the existing identity index.
    expected_files = {str(Path(row["image"]["arrow_file"]).resolve()) for row in rows}
    actual_files = {str(Path(item["filename"]).resolve()) for item in dataset.cache_files}
    if actual_files != expected_files:
        raise ValueError("HF selected different cache shards; expected the indexed official dataset revision")
    captions = dataset.select_columns(["caption"])
    for index, (row, example) in enumerate(zip(rows, captions)):
        if row.get("source_row_index") != index or caption_text(example["caption"]) != row["caption"]:
            raise ValueError(f"Official dataset order/caption differs from index at row {index}")
    return dataset
