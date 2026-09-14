"""Qwen-native denoising with DiT360's persistent padded inference state.

Original token count -> circular padding -> scheduler uses padded token count
-> denoise the padded state -> crop -> Qwen VAE unpack/denormalize/decode.
"""
import torch
from diffusers import QwenImagePipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput

from .circular import crop_columns, install_circular_padding, pad_columns


class QwenPanoPipeline(QwenImagePipeline):
    padding_columns = 0

    def enable_pano(self, columns):
        install_circular_padding(self.transformer, columns, mode="persistent")
        self.padding_columns = columns

    def prepare_latents(self, batch_size, num_channels_latents, height, width,
                        dtype, device, generator, latents=None):
        latents = super().prepare_latents(batch_size, num_channels_latents, height, width,
                                          dtype, device, generator, latents)
        if self.padding_columns:
            return pad_columns(latents, height // (self.vae_scale_factor * 2),
                               width // (self.vae_scale_factor * 2), self.padding_columns)
        return latents

    @torch.no_grad()
    def __call__(self, prompt=None, *, height=1024, width=2048,
                 output_type="pil", return_dict=True, **kwargs):
        if height <= 0 or height % (self.vae_scale_factor * 2) or width != 2 * height:
            raise ValueError("Use a 2:1 ERP with dimensions divisible by twice the VAE spatial scale")
        # The native pipeline still computes CFG and its resolution-dependent
        # scheduler shift. It now observes the padded sequence length.
        result = super().__call__(prompt=prompt, height=height, width=width,
                                  output_type="latent", return_dict=False, **kwargs)[0]
        if self.padding_columns:
            result = crop_columns(result, height // (self.vae_scale_factor * 2),
                                   width // (self.vae_scale_factor * 2), self.padding_columns)
        if output_type == "latent":
            image = result
        else:
            latents = self._unpack_latents(result, height, width, self.vae_scale_factor).to(self.vae.dtype)
            mean = torch.tensor(self.vae.config.latents_mean, device=latents.device,
                                dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
            inv_std = torch.tensor(self.vae.config.latents_std, device=latents.device,
                                   dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1).reciprocal()
            latents = latents / inv_std + mean
            decoded = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(decoded, output_type=output_type)
        self.maybe_free_model_hooks()
        return QwenImagePipelineOutput(images=image) if return_dict else (image,)
