"""Native Qwen flow-matching LoRA with DiT360 panorama supervision.

Single GPU or Accelerate DDP. Frozen base weights are NOT saved in checkpoints.
Checkpoint resume is at completed epoch boundaries; init-lora is weights-only.
"""
import argparse
import json
import math
from pathlib import Path

from .common import (DEFAULT_MODEL, DEFAULT_PROMPTS, atomic_json, model_snapshot,
                     require_versions, sha256, source_hashes, text_hash, versions)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("repo-panorama", "repo-mix", "paper", "custom"), default="repo-panorama")
    p.add_argument("--paper-lr-schedule", choices=("constant", "repo-step", "repo-epoch"))
    p.add_argument("--paper-mask-reduction", choices=("valid", "full"))
    p.add_argument("--geometry-backend", choices=("upstream", "periodic"))
    p.add_argument("--lr-schedule", choices=("constant", "repo-step", "repo-epoch"))
    p.add_argument("--vae-dtype", choices=("float32", "bfloat16"))
    p.add_argument("--max-grad-norm", type=float)
    p.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--panorama-manifest", type=Path, required=True)
    p.add_argument("--perspective-manifest", type=Path)
    p.add_argument("--heldout-manifest", type=Path, default=DEFAULT_PROMPTS,
                   help="Combined validation/test ID + scene_id manifest; defaults to current 1092 test prompts")
    p.add_argument("--allow-unverified-split", action="store_true")
    p.add_argument("--official-full-training-only", action="store_true",
                   help="Explicitly train the full official polished split despite known benchmark overlap; no independent-test claim")
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--init-lora", type=Path, help="Weights-only continuation, e.g. 512 -> 1024 height")
    p.add_argument("--height", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--accumulation-steps", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--rank", type=int)
    p.add_argument("--lora-alpha", type=int)
    p.add_argument("--lora-dropout", type=float)
    p.add_argument("--padding-columns", type=int)
    p.add_argument("--lambda-cube", type=float)
    p.add_argument("--lambda-yaw", type=float)
    p.add_argument("--lambda-seam", type=float)
    p.add_argument("--perspective-weight", type=float)
    p.add_argument("--schedule", choices=("panorama", "dit360-mix", "persistent-mix", "paper-hybrid"))
    p.add_argument("--geometry-start-epoch", type=int,
                   help="0-based; used for panorama/persistent-mix, ignored by released dit360-mix schedule")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--weighting-scheme", choices=("none", "logit_normal", "mode", "sigma_sqrt", "cosmap"))
    p.add_argument("--logit-mean", type=float)
    p.add_argument("--logit-std", type=float)
    p.add_argument("--mode-scale", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--vae-tiling", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()
    from .profiles import apply_profile
    apply_profile(args, p)
    if any(isinstance(value, float) and not math.isfinite(value) for value in vars(args).values()):
        p.error("Numeric parameters must be finite")
    if args.profile == "paper" and min(args.lambda_cube, args.lambda_yaw) <= 0:
        p.error("paper uses both geometric objectives; zero-weight ablations belong to custom")
    if args.height <= 0 or args.height % 16:
        p.error("height must be a positive multiple of 16; width is always 2 * height")
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.rank, args.lora_alpha, args.log_every) <= 0:
        p.error("epochs, batch, accumulation, LoRA rank/alpha and log-every must be positive")
    if args.padding_columns < 0 or args.padding_columns > args.height // 8:
        p.error("Invalid padding-columns")
    if min(args.lambda_cube, args.lambda_yaw, args.lambda_seam, args.perspective_weight,
           args.workers, args.warmup_steps or 0, args.geometry_start_epoch) < 0:
        p.error("Weights, workers, warmup and geometry-start-epoch cannot be negative")
    if not 0 <= args.lora_dropout < 1 or args.learning_rate <= 0:
        p.error("Invalid dropout or learning rate")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        p.error("max-grad-norm must be positive")
    if args.resume and args.init_lora:
        p.error("Choose resume OR init-lora")
    if args.official_full_training_only and args.profile != "repo-panorama":
        p.error("--official-full-training-only is for the repo-panorama full polished recipe")
    if (args.schedule == "panorama") == bool(args.perspective_manifest):
        p.error("panorama uses only panorama-manifest; mix schedules require perspective-manifest")
    return args


def main():
    args = parse_args()
    require_versions()
    import torch
    from accelerate import Accelerator
    from accelerate.utils import DistributedType, set_seed
    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImagePipeline, QwenImageTransformer2DModel
    from diffusers.training_utils import (_collate_lora_metadata, compute_density_for_timestep_sampling,
                                          compute_loss_weighting_for_sd3)
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader
    from .circular import install_circular_padding
    from .data import ERPTrainingDataset, collate, epoch_rows, training_rows
    from .losses import panorama_loss
    from .profiles import lr_multiplier

    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation_steps, mixed_precision="bf16")
    if accelerator.distributed_type not in (DistributedType.NO, DistributedType.MULTI_GPU):
        raise ValueError("This entry supports single GPU / DDP; FSDP and DeepSpeed need separate checkpoint handling")
    if accelerator.device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("Training requires a CUDA GPU with bfloat16 support")
    set_seed(args.seed, device_specific=True)
    allow_unverified = args.allow_unverified_split or args.profile.startswith("repo-")
    if args.official_full_training_only:
        provenance_path = args.panorama_manifest.with_suffix(".provenance.json")
        provenance = json.loads(provenance_path.read_text())
        if (provenance.get("dataset") != "Insta360-Research/Matterport3D_polished"
                or provenance.get("split_policy") != "official-full"
                or provenance.get("excluded_rows") != 0
                or provenance.get("source_rows") != provenance.get("training_rows")
                or provenance.get("manifest_sha256") != sha256(args.panorama_manifest)):
            raise ValueError("Official-full training requires an unfiltered, verified cached_data index")
    panoramas = training_rows(args.panorama_manifest, "panorama", args.heldout_manifest, allow_unverified, args.profile,
                             allow_overlap=args.official_full_training_only)
    perspectives = training_rows(args.perspective_manifest, "perspective", args.heldout_manifest,
                                 allow_unverified, args.profile) if args.perspective_manifest else []
    ids = [r["id"] for r in panoramas + perspectives]
    if len(ids) != len(set(ids)):
        raise ValueError("Panorama and perspective manifests must have globally unique IDs")
    if args.official_full_training_only and len(panoramas) != provenance["training_rows"]:
        raise ValueError("Official-full row count differs from index provenance")
    with accelerator.main_process_first():
        snapshot = model_snapshot(args.model, args.revision, args.local_files_only)
    cache_config = json.loads((args.text_cache / "cache_config.json").read_text())
    if cache_config["base_snapshot"] != snapshot or cache_config["versions"] != versions():
        raise ValueError("Text embeddings must use the same base snapshot and package versions as training")
    if not (args.text_cache / "complete.json").exists():
        raise ValueError("Finish cache_text before starting training")
    cache_complete = json.loads((args.text_cache / "complete.json").read_text())
    if cache_complete["config_sha256"] != sha256(args.text_cache / "cache_config.json"):
        raise ValueError("Text cache completion/config mismatch")
    for row in panoramas + perspectives:
        path = args.text_cache / (text_hash(row["caption"]) + ".safetensors")
        if not path.is_file():
            raise FileNotFoundError(f"Missing text cache for {row['id']}: {path}")
    # The upstream step/epoch warmup depends on the total epoch count. A resume
    # must not silently change it by allowing a new --epochs value.
    batches_per_epoch = len(panoramas) // args.batch_size
    computed_warmup = int(.05 * (batches_per_epoch * args.epochs // args.accumulation_steps))
    if args.warmup_steps is not None:
        computed_warmup = args.warmup_steps
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ("resume", "output")}
    config.update(base_snapshot=snapshot, width=2 * args.height, versions=versions(),
                  world_size=accelerator.num_processes, source_hashes=source_hashes(),
                  panorama_sha256=sha256(args.panorama_manifest),
                  perspective_sha256=sha256(args.perspective_manifest) if args.perspective_manifest else None,
                  heldout_sha256=sha256(args.heldout_manifest),
                  text_cache_config_sha256=sha256(args.text_cache / "cache_config.json"),
                  computed_warmup_steps=computed_warmup,
                  allow_missing_house_identity=allow_unverified,
                  actual_panorama_count=len(panoramas), actual_perspective_count=len(perspectives),
                  paper_perspective_count_matches_40000=len(perspectives) == 40000,
                  global_effective_batch=args.batch_size * args.accumulation_steps * accelerator.num_processes,
                  init_lora_sha256=sha256(args.init_lora / "adapter_resume.safetensors") if args.init_lora else None,
                  split_status="unverified" if any(not (r.get("scene_id") or r.get("scan_id")) for r in panoramas)
                  else "checked_against_supplied_heldout_houses")
    if args.official_full_training_only:
        config["split_status"] = "official_full_training_only_no_independent_test_claim"
        config["dataset_index_provenance"] = provenance
    if args.resume:
        old = json.loads((args.resume / "training_config.json").read_text())
        if old != config:
            raise ValueError("Resume configuration changed. For a new experiment, use --init-lora and a new output")
        if not (args.resume / "COMPLETE.json").exists():
            raise ValueError("Checkpoint incomplete; use the previous completed epoch checkpoint")
    config_path = args.output / "training_config.json"
    # Validate on every rank before rank zero mutates the shared directory.
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError("Output must be empty for a new training run")
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output belongs to another experiment")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(config_path, config)
    accelerator.wait_for_everyone()
    vae = AutoencoderKLQwenImage.from_pretrained(snapshot, subfolder="vae", torch_dtype=getattr(torch, args.vae_dtype),
                                                local_files_only=True).requires_grad_(False).eval().to(accelerator.device)
    if args.vae_tiling:
        vae.enable_tiling()
    vae_scale = 2 ** len(vae.temperal_downsample)
    if vae_scale != 8:
        raise ValueError("This implementation expects Qwen-Image's 8x spatial VAE")
    mean = torch.tensor(vae.config.latents_mean, device=accelerator.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=accelerator.device).reshape(1, -1, 1, 1, 1)
    transformer = QwenImageTransformer2DModel.from_pretrained(snapshot, subfolder="transformer",
                      torch_dtype=torch.bfloat16, local_files_only=True).requires_grad_(False)
    if transformer.config.guidance_embeds or getattr(transformer, "zero_cond_t", False):
        raise ValueError("Use a non-distilled Qwen-Image T2I base, not an Edit or Lightning model")
    targets = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"]
    transformer.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.lora_alpha,
                            lora_dropout=args.lora_dropout, init_lora_weights="gaussian", target_modules=targets))
    for param in transformer.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    if args.init_lora:
        init = json.loads((args.init_lora / "pano_config.json").read_text())
        if any(init[k] != v for k, v in (("base_snapshot", snapshot), ("rank", args.rank),
                                         ("lora_alpha", args.lora_alpha), ("padding_columns", args.padding_columns))):
            raise ValueError("init-lora base, rank, alpha and padding must match")
        result = set_peft_model_state_dict(transformer, load_file(str(args.init_lora / "adapter_resume.safetensors")))
        if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
            raise ValueError("Incomplete or incompatible init-lora state")
    install_circular_padding(transformer, args.padding_columns)
    transformer.enable_gradient_checkpointing()
    transformer.train()
    params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, betas=(.9, .999), weight_decay=1e-5, eps=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda index: lr_multiplier(index, args.lr_schedule, args.epochs, computed_warmup))
    transformer, optimizer = accelerator.prepare(transformer, optimizer)
    # Step exactly once per actual optimizer update; do not multiply by GPU count.
    accelerator.register_for_checkpointing(lr_scheduler)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(snapshot, subfolder="scheduler", local_files_only=True)
    # Use the scheduler's native training timestep/sigma table, matching the
    # official Qwen LoRA example. Inference uses its native resolution shift.
    sigmas_table = scheduler.sigmas.to(accelerator.device)
    timesteps_table = scheduler.timesteps.to(accelerator.device)
    pano_config = dict(base_snapshot=snapshot, padding_columns=args.padding_columns,
                       rank=args.rank, lora_alpha=args.lora_alpha, height=args.height, width=args.height * 2,
                       max_sequence_length=cache_config["max_sequence_length"],
                       targets=targets, geometry=args.geometry_backend, profile=args.profile,
                       inference_padding_columns=1 if args.profile.startswith("repo-") else args.padding_columns,
                       inference_padding_mode="persistent", schema_version=2)

    def save_hook(models, weights, directory):
        if accelerator.is_main_process:
            state = {k: v.detach().cpu().contiguous() for k, v in
                     get_peft_model_state_dict(accelerator.unwrap_model(models[0])).items()}
            save_file(state, str(Path(directory) / "adapter_resume.safetensors"))
            QwenImagePipeline.save_lora_weights(directory, transformer_lora_layers=state,
                **_collate_lora_metadata({"transformer": accelerator.unwrap_model(models[0])}))
            atomic_json(Path(directory) / "pano_config.json", pano_config)
            atomic_json(Path(directory) / "training_config.json", config)
        weights.clear()  # Prevent saving the frozen 20B base.

    def load_hook(models, directory):
        model = accelerator.unwrap_model(models.pop())
        result = set_peft_model_state_dict(model, load_file(str(Path(directory) / "adapter_resume.safetensors")))
        if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
            raise ValueError("Incomplete or incompatible checkpoint LoRA state")

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)
    start_epoch, step = 0, 0
    if args.resume:
        accelerator.load_state(str(args.resume))
        progress = json.loads((args.resume / "COMPLETE.json").read_text())
        start_epoch, step = progress["completed_epochs"], progress["optimizer_step"]
    if start_epoch >= args.epochs:
        raise ValueError("Checkpoint already reached requested epochs; use --init-lora for a separate continuation stage")
    accelerator.print(f"Trainable LoRA parameters: {sum(p.numel() for p in params):,}; {config['split_status']}")

    def encode_cached(captions):
        tensors = [load_file(str(args.text_cache / (text_hash(c) + ".safetensors"))) for c in captions]
        length = max(t["embeddings"].shape[0] for t in tensors)
        embeddings = torch.zeros(len(tensors), length, tensors[0]["embeddings"].shape[1], dtype=torch.bfloat16)
        masks = torch.zeros(len(tensors), length, dtype=torch.bool)
        for i, t in enumerate(tensors):
            n = t["embeddings"].shape[0]
            embeddings[i, :n], masks[i, :n] = t["embeddings"], t["mask"]
        return embeddings.to(accelerator.device), masks.to(accelerator.device)

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        rows = epoch_rows(panoramas, perspectives, epoch, args.seed, args.schedule)
        dataset = ERPTrainingDataset(rows, args.height, args.augment, profile=args.profile)
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                            collate_fn=collate, pin_memory=True, generator=generator, drop_last=args.drop_last)
        loader = accelerator.prepare_data_loader(loader)
        loader.set_epoch(epoch)
        if not len(loader):
            raise ValueError("No training batches remain after distributed sharding/drop_last")
        accelerator.print(f"Epoch {epoch + 1}/{args.epochs}: panorama={len(panoramas)}, perspective={len(rows) - len(panoramas)}")
        for batch in loader:
            with accelerator.accumulate(transformer):
                with torch.no_grad():
                    z = vae.encode(batch["pixels"].to(vae.dtype).unsqueeze(2)).latent_dist.sample()
                    z = ((z.float() - mean) / std).to(torch.bfloat16)
                    embeddings, text_mask = encode_cached(batch["captions"])
                noise = torch.randn_like(z)
                u = compute_density_for_timestep_sampling(weighting_scheme=args.weighting_scheme,
                            batch_size=len(z), logit_mean=args.logit_mean, logit_std=args.logit_std,
                            mode_scale=args.mode_scale)
                indices = (u * scheduler.config.num_train_timesteps).long().clamp_max(len(timesteps_table) - 1).to(z.device)
                t = timesteps_table[indices]
                sigma = sigmas_table[indices].reshape(-1, 1, 1, 1, 1).to(z.dtype)
                noisy = (1 - sigma) * z + sigma * noise
                packed = QwenImagePipeline._pack_latents(noisy, len(z), z.shape[1], z.shape[3], z.shape[4])
                with accelerator.autocast():
                    prediction = transformer(hidden_states=packed, timestep=t / 1000,
                          encoder_hidden_states=embeddings, encoder_hidden_states_mask=text_mask,
                          img_shapes=[[(1, z.shape[3] // 2, z.shape[4] // 2)]] * len(z), return_dict=False)[0]
                prediction = QwenImagePipeline._unpack_latents(prediction, args.height, args.height * 2, vae_scale)
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigma)
                pano = torch.tensor([k == "panorama" for k in batch["kinds"]], device=z.device)
                loss, components = panorama_loss(prediction, noise - z, batch["mask"], pano,
                                                  weighting, epoch, args, noise=noise, clean=z)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss in {batch['ids']}")
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    if args.lr_schedule != "repo-epoch":
                        lr_scheduler.step()
                    step += 1
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients and step % args.log_every == 0:
                average = accelerator.gather(loss.detach().reshape(1)).mean().item()
                if accelerator.is_main_process:
                    record = dict(epoch=epoch, step=step, loss_last_microbatch_across_ranks=average,
                                  lr=lr_scheduler.get_last_lr()[0],
                                  rank0_last_microbatch={k: v.item() for k, v in components.items()})
                    with (args.output / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    print(record, flush=True)
        if args.lr_schedule == "repo-epoch":
            lr_scheduler.step()
        checkpoint = args.output / f"checkpoint-epoch{epoch + 1:03d}"
        if checkpoint.exists():
            raise ValueError(f"Checkpoint already exists; use a new output when branching a run: {checkpoint}")
        accelerator.wait_for_everyone()
        accelerator.save_state(str(checkpoint))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            atomic_json(checkpoint / "COMPLETE.json", dict(completed_epochs=epoch + 1, optimizer_step=step))
        accelerator.wait_for_everyone()
    accelerator.print(f"Finished. Inference LoRA: {checkpoint}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
