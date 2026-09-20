"""Index existing Hugging Face Arrow images without exporting or copying images.

No model imports. The index contains captions, identities and Arrow row pointers.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from .common import DEFAULT_PROMPTS, ROOT, caption_text, read_jsonl, sha256

DATASET = "Insta360-Research/Matterport3D_polished"


def resolve_arrow_cache(source, revision=None):
    source = Path(source).expanduser().resolve()
    if (source / "dataset_info.json").is_file():
        if revision and source.name != revision:
            raise ValueError("Arrow cache directory does not match requested revision")
        return source
    if source.parent.name == "snapshots":
        if revision and source.name != revision:
            raise ValueError("Hub snapshot does not match requested revision")
        revision, hub = source.name, source.parent.parent
    else:
        hub = source
        ref = hub / "refs/main"
        if revision is None and ref.is_file():
            revision = ref.read_text().strip()
        if revision is None:
            snapshots = [p.name for p in (hub / "snapshots").glob("*")
                         if p.is_dir() and re.fullmatch(r"[0-9a-f]{40}", p.name)]
            if len(snapshots) == 1:
                revision = snapshots[0]
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pass a cached Arrow directory or a Hub cache root with --dataset-revision")
    bases = [hub.parent.parent / "datasets"]
    if os.environ.get("HF_DATASETS_CACHE"):
        bases.insert(0, Path(os.environ["HF_DATASETS_CACHE"]))
    matches = set()
    for base in bases:
        matches.update(p.resolve() for p in
                       (base / "Insta360-Research___matterport3_d_polished").glob(f"*/*/{revision}")
                       if (p / "dataset_info.json").is_file())
    if len(matches) != 1:
        raise ValueError("Expected one existing prepared Arrow cache for this revision; "
                         "pass its dataset_info.json parent directory as --source. "
                         "This command never downloads or converts Parquet files.")
    return matches.pop()


def house_map(caption_root):
    mapping = {}
    for path in sorted(Path(caption_root).glob("*/blip3_stitched/*.txt")):
        view, house = path.stem, path.parents[1].name
        if view in mapping and mapping[view] != house:
            raise ValueError(f"Ambiguous panorama/house mapping: {view}")
        mapping[view] = house
    return mapping


def build_index(args):
    import pyarrow as pa
    import pyarrow.compute as pc
    cache = resolve_arrow_cache(args.source, args.dataset_revision)
    info = json.loads((cache / "dataset_info.json").read_text())
    if info.get("dataset_name") != "matterport3_d_polished" or set(info["features"]) != {"image", "caption"}:
        raise ValueError("Expected the official Matterport3D_polished image/caption cache")
    expected_rows = info["splits"]["train"]["num_examples"]
    shard_lengths = info["splits"]["train"].get("shard_lengths")
    files = sorted(cache.glob("*-train-*.arrow"))
    if not files or (shard_lengths and len(files) != len(shard_lengths)):
        raise ValueError("Arrow training shards are missing")
    mapping = house_map(args.caption_root)
    heldout = read_jsonl(args.heldout_manifest)
    houses = {r.get("scene_id") or r.get("scan_id") for r in heldout}
    if None in houses:
        raise ValueError("Every heldout row needs a house identity")
    views = {r["view_id"] for r in heldout if r.get("view_id")}
    if args.split_policy == "exclude-heldout-houses" and not mapping:
        raise ValueError("House exclusion requires the existing */blip3_stitched/*.txt metadata")
    records, inventory, seen = [], [], set()
    total = excluded = overlaps = 0
    for file_number, file in enumerate(files):
        stat = file.stat()
        item = dict(arrow_file=str(file), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        inventory.append(item)
        shard_rows = 0
        with pa.memory_map(str(file), "r") as source:
            reader = pa.ipc.open_stream(source)
            for batch_index, batch in enumerate(reader):
                images = batch.column(batch.schema.get_field_index("image"))
                captions = batch.column(batch.schema.get_field_index("caption")).to_pylist()
                paths = images.field("path").to_pylist()
                data = images.field("bytes")
                if images.null_count or data.null_count or (len(data) and pc.min(pc.binary_length(data)).as_py() <= 0):
                    raise ValueError(f"Missing embedded image bytes: {file}")
                for row_index, (path, caption) in enumerate(zip(paths, captions)):
                    total += 1
                    shard_rows += 1
                    match = re.fullmatch(r"([0-9a-f]{32})_clean_big_dc\.png", Path(path or "").name)
                    if not match or match.group(1) in seen:
                        raise ValueError("Unrecognized or duplicate official panorama UUID")
                    view = match.group(1)
                    seen.add(view)
                    house = mapping.get(view)
                    overlaps += int(view in views)
                    caption = caption_text(caption)
                    if args.split_policy == "exclude-heldout-houses":
                        if house is None:
                            raise ValueError(f"Cannot establish house identity for {view}; no silent inclusion")
                        if house in houses:
                            excluded += 1
                            continue
                    records.append(dict(id=f"{house}_{view}" if house else f"polished_{view}",
                                        image={**item, "batch_index": batch_index, "row_index": row_index},
                                        caption=caption, source=DATASET, source_row_index=total - 1,
                                        source_image_path=path, source_view_id=view, scene_id=house,
                                        split="train", orientation_invariant=False))
        if shard_lengths and shard_rows != shard_lengths[file_number]:
            raise ValueError(f"Arrow row count mismatch in {file}")
    if total != expected_rows or not records:
        raise ValueError("Incomplete or empty dataset")
    provenance = dict(dataset=DATASET, dataset_revision=cache.name, source_rows=total,
                      training_rows=len(records), excluded_rows=excluded, source_benchmark_uuid_overlap=overlaps,
                      split_policy=args.split_policy, arrow_files=inventory,
                      dataset_info_sha256=sha256(cache / "dataset_info.json"),
                      heldout_sha256=sha256(args.heldout_manifest),
                      house_map_sha256=hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest(),
                      image_exported=False, note="Original image bytes stay in existing Arrow cache.")
    manifest_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    provenance["manifest_sha256"] = hashlib.sha256(manifest_text.encode()).hexdigest()
    metadata = args.output.with_suffix(".provenance.json")
    metadata_text = json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    # Re-running the same job is safe; a changed split/source needs a new index.
    for path, expected in [(args.output, manifest_text), (metadata, metadata_text)]:
        if path.exists() and path.read_text() != expected:
            raise ValueError(f"Existing index differs; choose a new --output: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, content in [(args.output, manifest_text), (metadata, metadata_text)]:
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(content)
            tmp.replace(path)
    print(json.dumps({k: provenance[k] for k in ("source_rows", "training_rows", "excluded_rows",
                                                "source_benchmark_uuid_overlap", "split_policy", "image_exported")}))


class ArrowImages:
    """Read one image without retaining every visited shard in each worker.

    Memory maps share file-backed pages; they are not full dataset copies.
    Closing each reader bounds retained mappings, not total job/page-cache RAM.
    """

    def open(self, pointer):
        import io
        import pyarrow as pa
        from PIL import Image
        with pa.memory_map(pointer["arrow_file"], "r") as source:
            with pa.ipc.open_stream(source) as reader:
                for index, batch in enumerate(reader):
                    if index == pointer["batch_index"]:
                        images = batch.column(batch.schema.get_field_index("image"))
                        # Copy the selected encoded image before closing its map.
                        encoded = images[pointer["row_index"]].as_py()["bytes"]
                        break
                else:
                    raise IndexError(f"Arrow batch not found: {pointer['batch_index']}")
        return Image.open(io.BytesIO(encoded))


def main():
    from huggingface_hub.constants import HF_HUB_CACHE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path,
                   default=Path(HF_HUB_CACHE) / "datasets--Insta360-Research--Matterport3D_polished",
                   help="Existing Hub root/snapshot or Arrow directory; defaults to Hugging Face's configured cache")
    p.add_argument("--dataset-revision")
    p.add_argument("--caption-root", type=Path, default=ROOT / "benchmark_assets/Matterport3D_stitched_captions/mp3d_skybox")
    p.add_argument("--heldout-manifest", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--split-policy", choices=("exclude-heldout-houses", "official-full"), default="official-full")
    p.add_argument("--output", type=Path, required=True)
    build_index(p.parse_args())


if __name__ == "__main__":
    main()
