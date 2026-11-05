"""flashcodec — compress time-series KV cache tensors as "high-dimensional
multimedia image signals".

Two classic compression theories ported to Transformer KV cache management:
- mode="jpeg": blockwise 1D DCT-II over an orthonormal basis, lossy
  frequency-domain compression (low-frequency skeleton);
- mode="png": causal blockwise prediction deltas, lossless/near-lossless
  time-domain compression (anchor + Δ).

Tensor contract: KV layout (B, H, S, D), compression axis dim=-2 (seq axis).
"""

from .attention import block_basis, codec_decode_attention
from .calibrate import calibrate_channel_scales, channel_scale_from_kv
from .cache import HAS_HF_CACHE, CodecKVCache, FlashCodecCache, \
    FlashCodecLayer, make_codec
from .codecs import JPEGCodec, PNGCodec, dct_basis, dct_compress, \
    dct_decompress, dequantize_coeffs, pack_coeffs, png_filter, \
    png_reconstruct, quantize_coeffs, unpack_coeffs
from .patch import apply_flash_codec, find_attention_modules, \
    remove_flash_codec, undo_flash_codec
from .patch_attention import apply_codec_attention, codec_attention_stats, \
    remove_codec_attention
from .storage import load_archive, save_archive

__version__ = "1.2.0"

_EXPORT_API = ("export_model", "load_pretrained", "read_manifest", "apply_export")


def __getattr__(name):
    # `flashcodec.export` is imported lazily: importing it eagerly makes
    # `python -m flashcodec.export` a double import (runpy warning).
    if name in _EXPORT_API:
        from . import export as _export
        return getattr(_export, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "apply_flash_codec", "remove_flash_codec", "undo_flash_codec",
    "find_attention_modules",
    "apply_codec_attention", "remove_codec_attention", "codec_attention_stats",
    "CodecKVCache", "FlashCodecCache", "FlashCodecLayer", "make_codec",
    "HAS_HF_CACHE", "JPEGCodec", "PNGCodec",
    "codec_decode_attention", "block_basis",
    "dct_basis", "dct_compress", "dct_decompress",
    "quantize_coeffs", "dequantize_coeffs", "pack_coeffs", "unpack_coeffs",
    "calibrate_channel_scales", "channel_scale_from_kv",
    "export_model", "load_pretrained", "read_manifest", "apply_export",
    "save_archive", "load_archive",
    "png_filter", "png_reconstruct",
    "__version__",
]
