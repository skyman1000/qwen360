"""Resumable Qwen zero-shot / adapted ERP benchmark. Run from project root."""
import argparse
import json
import math
import time
from pathlib import Path

from .common import (DEFAULT_MODEL, DEFAULT_PROMPTS, ERP_PREFIX, ROOT, atomic_json,
                     check_ids, model_snapshot, read_jsonl, require_versions,
                     sample_seed, sha256, source_hashes, text_hash, versions, write_jsonl)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--output", type=Path, default=ROOT / "qwen_pano/outputs/zero_shot_2512")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--true-cfg-scale", type=float, default=4.)
    p.add_argument("--negative-prompt", default=" ")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed-mode", choices=("per-id", "shared"), default="per-id")
    p.add_argument("--prompt-mode", choices=("verbatim", "erp-prefix"), default="verbatim")
    p.add_argument("--max-sequence-length", type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--offload", choices=("none", "model", "sequential"), default="model")
    p.add_argument("--vae-tiling", action="store_true")
    p.add_argument("--lora", type=Path, help="A checkpoint/final directory written by qwen_pano.train")
    p.add_argument("--padding-columns", type=int, help="Default: 0 for zero-shot, saved setting for LoRA")
    p.add_argument("--padding-mode", choices=("temporary", "persistent"), help="Default: saved checkpoint semantics")
    p.add_argument("--limit", type=int, help="First N samples; use a separate output for this pilot")
    return p.parse_args()


def main():
    args = parse_args()
    if args.width != 2 * args.height or args.height <= 0 or args.height % 16:
        raise ValueError("Request an exact 2:1 ERP with dimensions divisible by 16")
    if args.steps <= 0 or not 0 <= args.seed < 2 ** 63:
        raise ValueError("Invalid steps, sequence length or base seed")
    if not math.isfinite(args.true_cfg_scale) or args.true_cfg_scale <= 0:
        raise ValueError("true-cfg-scale must be positive and finite")
    rows = read_jsonl(args.prompts)
    check_ids(rows)
    if any(not isinstance(r.get("prompt"), str) or not r["prompt"].strip() for r in rows):
        raise ValueError("Each row needs a nonempty prompt")
    if args.limit is not None:
        if not 0 < args.limit <= len(rows):
            raise ValueError("limit must be in [1, manifest length]")
        rows = rows[:args.limit]
    require_versions()
    import torch
    from PIL import Image
    from diffusers import QwenImagePipeline
    from .circular import install_circular_padding

    if torch.device(args.device).type != "cuda":
        raise ValueError("This inference entry requires CUDA")
    torch.cuda.set_device(torch.device(args.device))
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This inference entry requires a CUDA GPU with bfloat16 support")
    snapshot = model_snapshot(args.model, args.revision, args.local_files_only)
    index = json.loads((Path(snapshot) / "model_index.json").read_text())
    if index.get("_class_name") != "QwenImagePipeline":
        raise ValueError("This benchmark supports Qwen-Image T2I, not Edit or other architectures")
    adapter_info = None
    padding = args.padding_columns
    padding_mode = args.padding_mode
    if args.lora:
        adapter_info = json.loads((args.lora / "pano_config.json").read_text())
        if adapter_info["base_snapshot"] != snapshot:
            raise ValueError("LoRA base snapshot differs; select the exact training --model snapshot")
        if padding is None:
            padding = adapter_info.get("inference_padding_columns", adapter_info["padding_columns"])
        if padding_mode is None:
            padding_mode = adapter_info.get("inference_padding_mode", "temporary")
        if args.max_sequence_length is None:
            args.max_sequence_length = adapter_info.get("max_sequence_length", 512)
    elif padding not in (None, 0) or padding_mode is not None:
        raise ValueError("Zero-shot uses the unmodified Qwen pipeline (padding=0)")
    if args.max_sequence_length is None:
        args.max_sequence_length = 512
    if not 1 <= args.max_sequence_length <= 1024:
        raise ValueError("max-sequence-length must lie in [1, 1024]")
    padding_mode = padding_mode or "none"
    padding = padding or 0
    if padding < 0 or padding > args.width // 16:
        raise ValueError("Invalid padding columns")
    run = dict(model=args.model, base_snapshot=snapshot, revision=args.revision,
               width=args.width, height=args.height, steps=args.steps,
               true_cfg_scale=args.true_cfg_scale, negative_prompt=args.negative_prompt,
               base_seed=args.seed, seed_mode=args.seed_mode, prompt_mode=args.prompt_mode,
               prompt_prefix=ERP_PREFIX if args.prompt_mode == "erp-prefix" else "",
               prompts_sha256=sha256(args.prompts), count=len(rows),
               selected_ids_sha256=text_hash("\n".join(r["id"] for r in rows)),
               max_sequence_length=args.max_sequence_length, padding_columns=padding,
               padding_mode=padding_mode,
               dtype="bfloat16", device=args.device, offload=args.offload, vae_tiling=args.vae_tiling,
               lora_sha256=sha256(args.lora / "pytorch_lora_weights.safetensors") if args.lora else None,
               adapter_info=adapter_info, versions=versions(), source_hashes=source_hashes(),
               seed_derivation="sha256(utf8(base_seed:sample_id)) first8 big-endian modulo 2**63")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = out / "generation_config.json"
    # Single writer per output directory. An interrupted process may leave the
    # lock; verify its PID/job ended before manually removing it.
    lock = out / ".generation.lock"
    with lock.open("x") as handle:
        import os
        handle.write(json.dumps(dict(pid=os.getpid(), slurm_job=os.getenv("SLURM_JOB_ID"))))
    pipe = None
    generated = []
    try:
        if cfg.exists():
            if json.loads(cfg.read_text()) != run:
                raise ValueError("Run configuration changed; choose a NEW output directory")
        elif any(p != lock for p in out.iterdir()):
            raise ValueError("Nonempty output without generation_config.json")
        else:
            atomic_json(cfg, run)
        for i, row in enumerate(rows, 1):
            seed = sample_seed(args.seed, row["id"], args.seed_mode)
            actual_prompt = run["prompt_prefix"] + row["prompt"]
            dest = out / f'{row["id"]}_seed{seed}.png'
            sidecar = dest.with_suffix(".json")
            # prompt stays the baseline caption for the shared evaluator.
            record = {**row, "prompt": row["prompt"], "generation_prompt": actual_prompt,
                      "seed": seed, "image": dest.name, "run_sha256": sha256(cfg)}
            if dest.exists() and sidecar.exists():
                old = json.loads(sidecar.read_text())
                if any(old.get(k) != v for k, v in record.items()) or old["image_sha256"] != sha256(dest):
                    raise ValueError(f"Output metadata/content mismatch: {dest}")
                with Image.open(dest) as im:
                    if im.size != (args.width, args.height):
                        raise ValueError(f"Wrong dimensions: {dest}")
                    im.verify()
                record = old
                print(f'[{i}/{len(rows)}] skip {row["id"]}', flush=True)
            else:
                if pipe is None:
                    from .pipeline import QwenPanoPipeline
                    pipeline_class = QwenPanoPipeline if padding_mode == "persistent" else QwenImagePipeline
                    pipe = pipeline_class.from_pretrained(snapshot, torch_dtype=torch.bfloat16, local_files_only=True)
                    if args.lora:
                        pipe.load_lora_weights(str(args.lora), weight_name="pytorch_lora_weights.safetensors")
                    if padding_mode == "persistent":
                        pipe.enable_pano(padding)
                    else:
                        install_circular_padding(pipe.transformer, padding)
                    if args.vae_tiling:
                        pipe.vae.enable_tiling()
                    if args.offload == "none":
                        pipe.to(args.device)
                    elif args.offload == "model":
                        pipe.enable_model_cpu_offload(device=args.device)
                    else:
                        pipe.enable_sequential_cpu_offload(device=args.device)
                print(f'[{i}/{len(rows)}] generate {row["id"]}, seed={seed}', flush=True)
                torch.cuda.reset_peak_memory_stats(args.device)
                start = time.perf_counter()
                image = pipe(prompt=actual_prompt, negative_prompt=args.negative_prompt,
                             width=args.width, height=args.height, num_inference_steps=args.steps,
                             true_cfg_scale=args.true_cfg_scale, max_sequence_length=args.max_sequence_length,
                             generator=torch.Generator(device=args.device).manual_seed(seed)).images[0]
                torch.cuda.synchronize(args.device)
                if image.size != (args.width, args.height):
                    raise ValueError("Pipeline changed dimensions; no post-generation stretching is allowed")
                record.update(elapsed_seconds=time.perf_counter() - start,
                              peak_allocated_gib=torch.cuda.max_memory_allocated(args.device) / 2 ** 30)
                tmp = dest.with_suffix(".png.tmp")
                image.save(tmp, format="PNG")
                tmp.replace(dest)
                record["image_sha256"] = sha256(dest)
                atomic_json(sidecar, record)
            generated.append(record)
        write_jsonl(out / "generated.jsonl", generated)
    finally:
        lock.unlink()
    print(f"Complete: {len(generated)} samples; {out / 'generated.jsonl'}")


if __name__ == "__main__":
    main()
