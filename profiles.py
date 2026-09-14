"""Separate released launch configurations from incomplete paper specifications.

Profiles describe the transferable recipe, not a claim of bitwise FLUX/Qwen parity.
See AUDIT.md for the unavoidable backend differences and unavailable paper details.
"""
UPSTREAM_COMMIT = "3779fe7965473f6824994c663a0ae7a76bc7aafa"

REPO_PANORAMA = dict(height=1024, epochs=25, batch_size=1, accumulation_steps=4,
    learning_rate=5e-5, rank=64, lora_alpha=64, lora_dropout=.05,
    padding_columns=1, lambda_cube=.5, lambda_yaw=.5, lambda_seam=0.,
    perspective_weight=.5, schedule="panorama", geometry_start_epoch=2,
    augment=True, weighting_scheme="none", logit_mean=0., logit_std=1., mode_scale=1.29,
    seed=0, geometry_backend="upstream", mask_reduction="valid", lr_schedule="repo-step",
    drop_last=True, max_grad_norm=None, vae_dtype="float32")
REPO_MIX = {**REPO_PANORAMA, "accumulation_steps": 5, "padding_columns": 0,
            "lambda_cube": 0., "lambda_yaw": 0., "schedule": "dit360-mix",
            "lr_schedule": "repo-epoch"}


def apply_profile(args, parser):
    if args.profile == "repo-panorama":
        recipe = REPO_PANORAMA
    elif args.profile == "repo-mix":
        recipe = REPO_MIX
    elif args.profile == "paper":
        # The paper never specifies these choices; the caller must declare
        # them rather than having an implementation silently invent them.
        required = ("lambda_cube", "lambda_yaw", "paper_lr_schedule", "paper_mask_reduction")
        if any(getattr(args, key) is None for key in required):
            parser.error("paper needs explicit --lambda-cube, --lambda-yaw, --paper-lr-schedule, "
                         "--paper-mask-reduction; their exact settings are not disclosed in the paper")
        recipe = {**REPO_PANORAMA, "epochs": 20, "accumulation_steps": 3,
                  "learning_rate": 2e-5, "schedule": "paper-hybrid", "geometry_start_epoch": 0,
                  "lambda_cube": args.lambda_cube, "lambda_yaw": args.lambda_yaw,
                  "perspective_weight": 1., "lr_schedule": args.paper_lr_schedule,
                  "mask_reduction": args.paper_mask_reduction}
    else:
        recipe = {**REPO_PANORAMA, "augment": False, "geometry_backend": "periodic",
                  "lr_schedule": "constant", "drop_last": False, "max_grad_norm": 1., "vae_dtype": "bfloat16"}
    for key, default in recipe.items():
        supplied = getattr(args, key, None)
        if args.profile != "custom" and supplied is not None and supplied != default:
            parser.error(f"{args.profile} fixes --{key.replace('_', '-')}={default!r}; "
                         "use --profile custom for an ablation")
        setattr(args, key, default if supplied is None else supplied)
    args.workers = 25 if args.workers is None else args.workers
    if args.profile != "custom" and args.warmup_steps is not None:
        parser.error("The selected profile computes warmup from the upstream schedule; no fixed --warmup-steps")
    if args.profile != "custom" and args.vae_tiling:
        parser.error("VAE tiling was not enabled upstream; use --profile custom for this change")
    args.upstream_commit = UPSTREAM_COMMIT


def lr_multiplier(index, schedule, epochs, warmup_steps):
    if schedule == "repo-epoch":
        warmup_epochs = max(1, epochs // 10)
        return .2 + .8 * (index / warmup_epochs) if index < warmup_epochs else 1.
    if schedule == "repo-step":
        if warmup_steps == 0:
            return 1.
        return min(1., index / max(1, warmup_steps))
    return 1.
