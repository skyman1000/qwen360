"""DiT360 position-aware circular token padding for native QwenImage.

Repeat both boundary tokens and their ORIGINAL complex RoPE frequencies.
Training uses temporary padding; inference can keep the padded state for all
scheduler steps, as in the upstream DiT360 pipeline. Text RoPE uses the original
image shape. No changes to parameter names or attention kernels.
"""
import torch


def image_hw(shapes):
    if not shapes:
        raise ValueError("img_shapes required")
    normalized = []
    for sample in shapes:
        if isinstance(sample[0], (list, tuple)):
            if len(sample) != 1:
                raise ValueError("Only single-image T2I is supported; Edit models need separate handling")
            sample = sample[0]
        f, h, w = sample
        if f != 1 or w != 2 * h:
            raise ValueError(f"Expected a single 2:1 ERP token grid, got {sample}")
        normalized.append((h, w))
    if len(set(normalized)) != 1:
        raise ValueError("Variable image shapes in a batch are unsupported")
    return normalized[0]


def pad_columns(x, h, w, n):
    if not 0 < n <= w or x.shape[-2] != h * w:
        raise ValueError("Invalid circular padding or token count")
    grid = x.reshape(*x.shape[:-2], h, w, x.shape[-1])
    return torch.cat((grid[..., -n:, :], grid, grid[..., :n, :]), dim=-2).flatten(-3, -2)


def crop_columns(x, h, w, n):
    grid = x.reshape(*x.shape[:-2], h, w + 2 * n, x.shape[-1])
    return grid[..., n:n + w, :].flatten(-3, -2)


def install_circular_padding(transformer, columns, mode="temporary"):
    if mode not in ("temporary", "persistent"):
        raise ValueError("Unknown circular padding mode")
    if columns < 0:
        raise ValueError("padding columns must be nonnegative")
    if getattr(transformer, "_pano_padding_installed", False):
        raise ValueError("Circular padding already installed")
    if columns == 0:
        return
    if getattr(transformer, "zero_cond_t", False) or transformer.config.guidance_embeds:
        raise ValueError("Only non-distilled Qwen-Image / Qwen-Image-2512 T2I checkpoints supported")

    def before(module, args, kwargs):
        if args or "hidden_states" not in kwargs:
            raise ValueError("Call the Qwen transformer using keyword arguments")
        if kwargs.get("controlnet_block_samples") is not None:
            raise ValueError("ControlNet residuals require their own padding implementation")
        h, w = image_hw(kwargs["img_shapes"])
        kwargs = dict(kwargs)
        if mode == "temporary":
            kwargs["hidden_states"] = pad_columns(kwargs["hidden_states"], h, w, columns)
        elif kwargs["hidden_states"].shape[1] != h * (w + 2 * columns):
            raise ValueError("Persistent mode expects already padded scheduler latents")
        return args, kwargs

    def rope_after(module, args, kwargs, output):
        shapes = args[0] if args else kwargs["video_fhw"]
        h, w = image_hw(shapes)
        image_rope, text_rope = output
        return pad_columns(image_rope, h, w, columns), text_rope

    def after(module, args, kwargs, output):
        if mode == "persistent":
            return output
        h, w = image_hw(kwargs["img_shapes"])
        if isinstance(output, tuple):
            return (crop_columns(output[0], h, w, columns), *output[1:])
        output.sample = crop_columns(output.sample, h, w, columns)
        return output

    transformer.register_forward_pre_hook(before, with_kwargs=True)
    transformer.pos_embed.register_forward_hook(rope_after, with_kwargs=True)
    transformer.register_forward_hook(after, with_kwargs=True)
    transformer._pano_padding_installed = True
    transformer._pano_padding_mode = mode
