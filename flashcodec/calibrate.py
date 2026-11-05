"""Channel-profile calibration for the jpeg codec.

The per-(block, frequency) scale is shared by all ``head_dim`` channels, but
per-channel coefficient magnitudes differ by orders of magnitude, so a single
scale wastes most of the quantizer range on a handful of loud channels.  A
*static* ``(keep_freqs, head_dim)`` profile ``g`` folded into the step
(``step = scale * g``) removes that mismatch:

    Qwen3-0.6B post-RoPE KV, 8-bit, half the tokens used to calibrate:
      layer 0  K   relative RMSE  0.0224 -> 0.0059   (q95 profile)
      layer 14 V   relative RMSE  0.0094 -> 0.0065

which is what makes a 7-bit archive (12.5% smaller) as accurate as an
uncalibrated 8-bit one.  K and V profiles differ a lot (sharing one table is
worse than no table on V), so both are returned separately.

Usage::

    ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    calib = calibrate_channel_scales(model, ids, keep_freqs=8)
    apply_flash_codec(model, mode="jpeg", keep_freqs=8, quant_bits=7,
                      per_layer=calib)
"""

import torch

from .codecs import dct_compress

__all__ = ["channel_scale_from_kv", "calibrate_channel_scales"]


def _profile(x, block_size, keep_freqs, quantile):
    """(B,H,S,D) -> (keep_freqs, D) fp32 profile, normalised to mean 1."""
    r = block_size
    n = x.shape[-2] // r
    if n == 0:
        raise ValueError(f"need at least {r} tokens to calibrate, got "
                         f"{x.shape[-2]}")
    x = x[..., : n * r, :]
    coef = dct_compress(x, r, keep_freqs)                 # (B,H,n,k,D)
    D = coef.shape[-1]
    rows = coef.abs().reshape(-1, keep_freqs, D).float()
    g = torch.quantile(rows, quantile, dim=0)             # (k,D)
    g = g / g.mean().clamp_min(1e-12)
    # a dead channel would otherwise blow up on the first real token
    return g.clamp_min(1e-3).cpu()


def channel_scale_from_kv(key_states, value_states, block_size=8,
                          keep_freqs=8, quantile=0.95):
    """Profiles for one layer: ``(g_key, g_value)``, each (keep_freqs, D)."""
    gk = _profile(key_states, block_size, keep_freqs, quantile)
    gv = _profile(value_states, block_size, keep_freqs, quantile)
    return gk, gv


def calibrate_channel_scales(model, input_ids, block_size=8, keep_freqs=8,
                             quantile=0.95, max_tokens=None):
    """Run one forward pass and build a ``per_layer`` dict for
    ``apply_flash_codec`` / ``FlashCodecCache.from_config``.

    ``input_ids``: (1, S) token ids on the model's device (S >= block_size;
    1-4k tokens is plenty — the profile is a few hundred numbers per layer).
    Returns ``{layer_idx: {"channel_scale": g_k, "channel_scale_v": g_v}}`` with
    CPU fp32 tensors; layers without a KV cache (linear attention) are skipped.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be (1, S), got {tuple(input_ids.shape)}")
    if max_tokens is not None:
        input_ids = input_ids[:, :max_tokens]
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    cache = out.past_key_values
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError("calibration needs a transformers 5.x Cache with "
                        ".layers (got %r)" % type(cache))
    scales = {}
    for idx, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None or keys.numel() == 0:
            continue
        gk, gv = channel_scale_from_kv(keys, values, block_size=block_size,
                                       keep_freqs=keep_freqs,
                                       quantile=quantile)
        scales[idx] = {"channel_scale": gk, "channel_scale_v": gv}
    return scales
