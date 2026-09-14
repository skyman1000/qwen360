"""Cache frozen native Qwen text embeddings separately from transformer training."""
import argparse
import json
from pathlib import Path

from .common import (DEFAULT_MODEL, atomic_json, caption_text, model_snapshot, read_jsonl,
                     require_versions, sha256, text_hash, versions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifests", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-sequence-length", type=int, default=512)
    args = p.parse_args()
    if not 1 <= args.max_sequence_length <= 1024:
        raise ValueError("max-sequence-length must lie in [1, 1024]")
    prompts = sorted({caption_text(r["caption"]) for path in args.manifests for r in read_jsonl(path)})
    if any(not isinstance(s, str) or not s.strip() for s in prompts):
        raise ValueError("Missing or empty caption")
    require_versions()
    import torch
    from diffusers import QwenImagePipeline
    from safetensors.torch import load_file, save_file

    snapshot = model_snapshot(args.model, args.revision, args.local_files_only)
    cfg = dict(base_snapshot=snapshot, max_sequence_length=args.max_sequence_length,
               dtype="bfloat16", versions=versions(),
               manifests={str(p.resolve()): sha256(p) for p in args.manifests})
    args.output.mkdir(parents=True, exist_ok=True)
    meta = args.output / "cache_config.json"
    if meta.exists():
        if json.loads(meta.read_text()) != cfg:
            raise ValueError("Text cache config changed; use a new directory")
    elif any(args.output.iterdir()):
        raise ValueError("Nonempty cache directory without provenance")
    else:
        atomic_json(meta, cfg)
    pipe = None
    for i, prompt in enumerate(prompts, 1):
        path = args.output / (text_hash(prompt) + ".safetensors")
        if path.exists():
            load_file(str(path))  # Detect damaged files before accepting a resume.
            continue
        if pipe is None:
            pipe = QwenImagePipeline.from_pretrained(snapshot, transformer=None, vae=None,
                                                     torch_dtype=torch.bfloat16, local_files_only=True)
            pipe.text_encoder.requires_grad_(False).eval().to(args.device)
        with torch.no_grad():
            embeddings, mask = pipe.encode_prompt(prompt, device=torch.device(args.device),
                                                  max_sequence_length=args.max_sequence_length)
        if mask is None:
            mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
        tmp = path.with_suffix(".tmp")
        save_file({"embeddings": embeddings[0].cpu().contiguous(),
                   "mask": mask[0].bool().cpu().contiguous()}, str(tmp), metadata={"prompt": prompt})
        tmp.replace(path)
        print(f"[{i}/{len(prompts)}] cached", flush=True)
    atomic_json(args.output / "complete.json", dict(count=len(prompts), config_sha256=sha256(meta)))


if __name__ == "__main__":
    main()
