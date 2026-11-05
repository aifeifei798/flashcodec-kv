"""Archive serialisation: the entropy-coding layer for disk / transport.

The hot path stores coefficients as a (possibly bit-packed) int8 tensor because
decoding attention reads it directly every step — an entropy coder there would
have to be re-run per step and would cost more than it saves.  Off the hot path
(save a long context, ship a cache to another machine) the picture is the
opposite: the symbols have ~6.8 bits of entropy at an 8-bit quantizer, so a
general-purpose coder buys ~10-15% for free.

What is written is the *logical* archive: packed coefficients are unpacked to
int8 first (bit-shifted streams compress worse), every part of a stream is
concatenated and compressed as one zlib/zstd blob.  Loading restores the
in-memory representation the codec wants (re-packing when ``quant_bits < 8``),
so a save/load round trip is bitwise exact.

Usage::

    stats = save_archive(cache, "ctx.fckv")
    cache = load_archive("ctx.fckv", config=model.config)
"""

import json
import struct
import zlib

import torch

from .cache import HAS_HF_CACHE, CodecKVCache, FlashCodecCache
from .codecs import pack_coeffs, unpack_coeffs

try:                                    # optional, no hard dependency
    import zstandard as _zstd
except Exception:                       # pragma: no cover
    _zstd = None

try:                                    # numpy is optional for torch
    import numpy as _np
except Exception:                       # pragma: no cover
    _np = None

__all__ = ["save_archive", "load_archive"]

_MAGIC = b"FCKV"
_VERSION = 1


# ------------------------------------------------------------------ helpers ---

def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _tensor_bytes(t):
    """Tensor -> raw bytes (contiguous, dtype-agnostic)."""
    t = t.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    if _np is not None:
        return t.numpy().tobytes()
    return bytes(t.tolist())          # numpy-free fallback (slower)


def _tensor_from_bytes(buf, shape, dtype_name):
    dtype = getattr(torch, dtype_name)
    n = 1
    for s in shape:
        n *= s
    t = torch.frombuffer(bytearray(buf), dtype=dtype, count=n)
    return t.reshape(shape)


def _compress(buf, level):
    if _zstd is not None:
        return _zstd.ZstdCompressor(level=level).compress(buf)
    return zlib.compress(buf, min(9, max(1, level)))


def _decompress(buf):
    if _zstd is not None and buf[:4] == b"\x28\xb5\x2f\xfd":
        return _zstd.ZstdDecompressor().decompress(buf)
    return zlib.decompress(buf)


def _stream_parts(stream):
    """Ordered (name, tensor, stored_bits) for one stream's valid region."""
    codec = stream.codec
    n = stream.n_blocks
    parts = []
    if codec.name == "jpeg":
        parts.append(("coeffs", stream._coeffs[:, :, :n], codec.quant_bits))
        if codec.quantized:
            parts.append(("scales", stream._scales[:, :, :n], 0))
        if codec.channel_scale is not None:
            parts.append(("channel_scale", codec.channel_scale, 0))
    else:
        parts.append(("deltas", stream._deltas[:, :, :n], 0))
        parts.append(("anchors", stream._anchors[:, :, :n], 0))
    if stream.raw is not None:
        parts.append(("raw", stream.raw, 0))
    return parts


def _codec_cfg(stream):
    codec = stream.codec
    cfg = {"mode": codec.name, "block_size": codec.block_size}
    if codec.name == "jpeg":
        cfg.update(keep_freqs=codec.keep_freqs, quant_bits=codec.quant_bits)
    else:
        cfg.update(keep_ratio=codec.keep_ratio)
    return cfg


# --------------------------------------------------------------------- save ---

def _iter_streams(cache):
    """Yield (layer_idx, 'keys'|'values', stream) for every codec stream."""
    if isinstance(cache, FlashCodecCache):
        for i, layer in enumerate(cache.layers):
            if hasattr(layer, "_key_stream"):
                yield i, "keys", layer._key_stream
                yield i, "values", layer._value_stream
    elif isinstance(cache, CodecKVCache):
        for i in sorted(cache._keys):
            yield i, "keys", cache._keys[i]
            yield i, "values", cache._values[i]
    else:
        raise TypeError(f"cannot serialise {type(cache)!r} — expected "
                        f"FlashCodecCache or CodecKVCache")


def save_archive(cache, path, level=6):
    """Write the archive to ``path`` (zlib, or zstd when importable).

    Returns ``{"raw_bytes", "stored_bytes", "ratio"}`` for the payload.
    """
    header = {"format": "flashcodec-archive", "version": _VERSION,
              "mem": cache.__class__.__name__, "layers": {}}
    payload = bytearray()
    raw_total = 0
    for idx, side, stream in _iter_streams(cache):
        if stream.n_blocks == 0 and stream.raw is None:
            continue
        buf = b""
        meta = []
        for name, t, bits in _stream_parts(stream):
            if bits and bits < 8:          # store the logical int8 stream
                t = unpack_coeffs(t, bits, head_dim=stream.head_dim)
            b = _tensor_bytes(t)
            meta.append({"name": name, "shape": list(t.shape),
                         "dtype": _dtype_name(t.dtype), "nbytes": len(b),
                         "bits": bits})
            buf += b
        blob = _compress(buf, level)
        entry = {
            "dtype": _dtype_name(stream.dtype),
            "head_dim": stream.head_dim,
            "n_archived": stream.n_archived,
            "n_blocks": stream.n_blocks,
            "parts": meta,
            "offset": len(payload),
            "clen": len(blob),
        }
        payload += blob
        raw_total += len(buf)
        layer = header["layers"].setdefault(str(idx), {"cfg": _codec_cfg(stream)})
        layer[side] = entry
    header["payload_bytes"] = len(payload)
    hjson = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(_MAGIC)
        fh.write(struct.pack("<II", _VERSION, len(hjson)))
        fh.write(hjson)
        fh.write(payload)
    return {"raw_bytes": raw_total, "stored_bytes": len(payload) + len(hjson),
            "ratio": (len(payload) + len(hjson)) / max(1, raw_total)}


# --------------------------------------------------------------------- load ---

def _restore_stream(stream, entry, payload, device=None):
    blob = _decompress(payload[entry["offset"]:entry["offset"] + entry["clen"]])
    parts = {}
    pos = 0
    for p in entry["parts"]:
        buf = blob[pos:pos + p["nbytes"]]
        pos += p["nbytes"]
        t = _tensor_from_bytes(buf, p["shape"], p["dtype"])
        if device is not None:
            t = t.to(device)          # re-pack on the target device (GPU is ~100x faster)
        if p["name"] == "coeffs" and stream.codec.packed:
            t = pack_coeffs(t, stream.codec.quant_bits).contiguous()
        parts[p["name"]] = t
    n = entry["n_blocks"]
    r = stream.block_size
    codec = stream.codec
    stream.dtype = getattr(torch, entry["dtype"])
    stream.head_dim = entry["head_dim"]
    stream.n_archived = entry["n_archived"]
    stream.n_blocks = n
    any_t = next(iter(parts.values()))
    stream.meta = (any_t.shape[0], any_t.shape[1], stream.head_dim,
                   stream.dtype.itemsize)
    if codec.name == "jpeg":
        stream._coeffs = parts["coeffs"]
        stream._scales = parts.get("scales")
    else:
        stream._deltas = parts["deltas"]
        stream._anchors = parts["anchors"]
    stream.raw = parts.get("raw")
    if parts.get("channel_scale") is not None:
        stream.codec.set_channel_scale(parts["channel_scale"].float())
    stream._capacity = max(n, 1)
    stream._rec = None
    stream._rec_blocks = 0
    stream._rec_capacity = 0


def _cfg_from_header(header, layers):
    """Global codec kwargs + per-layer overrides rebuilt from the file."""
    first = next(iter(layers.values()))
    cfg = dict(first["cfg"])
    per_layer = {int(idx): dict(entry["cfg"]) for idx, entry in layers.items()}
    return cfg, per_layer


def load_archive(path, config=None, device=None, materialize=False):
    """Read an archive written by :func:`save_archive`.

    With ``config`` (the model config) the result is a ``FlashCodecCache`` with
    exactly the layers that were saved; without it a ``CodecKVCache`` (single
    shared codec — per-layer channel profiles are then rejected, since silently
    dropping them would change the archive's numbers).

    ``materialize=True`` also builds each layer's reconstructed view eagerly.
    It is off by default: the view is ``S*D*2`` bytes per (layer, stream), i.e.
    several GB for a long context, and the next ``update``/``generate`` builds
    it anyway (the materialization-free path never needs it).
    """
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != _MAGIC:
            raise ValueError(f"{path}: not a flashcodec archive")
        version, hlen = struct.unpack("<II", fh.read(8))
        if version != _VERSION:
            raise ValueError(f"{path}: unsupported archive version {version}")
        header = json.loads(fh.read(hlen).decode("utf-8"))
        payload = fh.read()

    layers = {int(k): v for k, v in header["layers"].items()}
    if not layers:
        raise ValueError(f"{path}: archive has no layers")
    cfg, per_layer = _cfg_from_header(header, layers)

    if HAS_HF_CACHE and config is not None:
        cache = FlashCodecCache.from_config(config, per_layer=per_layer, **cfg)
        streams = {}
        for idx, layer in enumerate(cache.layers):
            if hasattr(layer, "_key_stream"):
                streams[idx] = (layer._key_stream, layer._value_stream)
        for idx, entry in layers.items():
            if idx not in streams:
                raise ValueError(f"layer {idx} in archive is not a codec layer "
                                 f"for this config")
            ks, vs = streams[idx]
            _restore_stream(ks, entry["keys"], payload, device)
            _restore_stream(vs, entry["values"], payload, device)
            layer = cache.layers[idx]
            layer.dtype, layer.device = ks.dtype, ks._coeffs.device
            if materialize:
                layer.is_initialized = True
                layer._rebuild_views()
    else:
        if len(layers) > 1:
            raise ValueError(
                "this archive has several codec layers; pass config= to "
                "rebuild a FlashCodecCache (a shared-codec CodecKVCache would "
                "change the archive's numbers)")
        cache = CodecKVCache(**cfg)
        idx, entry = next(iter(layers.items()))
        _restore_stream(cache._stream(cache._keys, idx), entry["keys"],
                        payload, device)
        _restore_stream(cache._stream(cache._values, idx), entry["values"],
                        payload, device)

    return cache
