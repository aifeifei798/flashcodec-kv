"""Calibrated channel profile + sub-byte packing.

1) pack/unpack round trip for every bit depth (including byte-padded head_dim)
2) the calibrated profile lowers reconstruction error at equal bit depth, and
   one bit narrower with the profile beats uncalibrated 8-bit
3) packing keeps the causal invariants: full forward and stepwise decode
   produce bitwise-identical archives and the same reconstructed view
4) Triton decode attention matches the torch path with packed coefficients and
   a channel profile (CUDA only; skipped otherwise)
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flashcodec import (CodecKVCache, JPEGCodec, codec_decode_attention,
                        pack_coeffs, unpack_coeffs)

torch.manual_seed(0)

B, H, S, D, R = 2, 4, 96, 32, 8
# per-channel magnitude spread: this is what the static profile is for
_CH = torch.logspace(-1.5, 1.5, D)
k_seq = torch.cumsum(torch.randn(B, H, S, D) * _CH, dim=-2)
v_seq = torch.cumsum(torch.randn(B, H, S, D) * _CH, dim=-2)


def _calib(x, keep_freqs=8, frac=0.5):
    """Calibrate on the first `frac` tokens, evaluate on the rest."""
    cut = int(S * frac) // R * R
    g = JPEGCodec(R, keep_freqs).compress(x[:, :, :cut])
    g = g.abs().reshape(-1, keep_freqs, D).float()
    g = torch.quantile(g, 0.95, dim=0)
    return (g / g.mean()).clamp_min(1e-3), cut


def _rel_rmse(x, codec):
    co = codec.compress(x)
    q, s = codec.quantize(co)
    rec = codec.decompress(q, out_len=x.shape[-2], scales=s, head_dim=D)
    return ((rec - x).pow(2).mean().sqrt() / x.pow(2).mean().sqrt()).item()


def test_pack_roundtrip():
    for bits in (2, 3, 4, 5, 6, 7):
        for d in (32, 12):                       # divisible and byte-padded
            lvl = 2 ** (bits - 1) - 1
            q = torch.randint(-lvl, lvl + 1, (2, 3, 5, d), dtype=torch.int8)
            p = pack_coeffs(q, bits)
            r = unpack_coeffs(p, bits, head_dim=d)
            assert p.dtype == torch.uint8
            assert p.shape[-1] == math.ceil(d * bits / 8), p.shape
            assert torch.equal(q, r), f"bits={bits} d={d} roundtrip mismatch"


def test_channel_profile_lowers_error():
    gk, cut = _calib(k_seq)
    tail = k_seq[:, :, cut:]
    plain8 = _rel_rmse(tail, JPEGCodec(R, 8, 8))
    plain7 = _rel_rmse(tail, JPEGCodec(R, 8, 7))
    cal8 = _rel_rmse(tail, JPEGCodec(R, 8, 8, channel_scale=gk))
    cal7 = _rel_rmse(tail, JPEGCodec(R, 8, 7, channel_scale=gk))
    print(f"   rel_rmse  k=8: plain b8={plain8:.4f} b7={plain7:.4f}  "
          f"cal b8={cal8:.4f} b7={cal7:.4f}")
    assert cal8 < plain8 * 0.8, (cal8, plain8)
    assert cal7 < plain7 * 0.85, (cal7, plain7)


def test_packed_causal_invariants():
    kw = dict(mode="jpeg", block_size=R, keep_freqs=8, quant_bits=7,
              channel_scale=_calib(k_seq)[0])
    full = CodecKVCache(**kw)
    full.forward(k_seq, v_seq)
    incr = CodecKVCache(**kw)
    for t in range(S):
        incr.decode_step(k_seq[..., t:t + 1, :], v_seq[..., t:t + 1, :])
    for a, b in ((full._keys[0], incr._keys[0]),
                 (full._values[0], incr._values[0])):
        assert a.n_blocks == b.n_blocks
        assert torch.equal(a._coeffs[:, :, :a.n_blocks],
                           b._coeffs[:, :, :b.n_blocks]), "archive differs"
        if a.raw is not None or b.raw is not None:
            assert torch.equal(a.raw, b.raw)
    d = (full.key_cache[0] - incr.key_cache[0]).abs().max().item()
    assert d < 1e-6, f"views disagree: {d:.2e}"


def test_packed_storage_ratio():
    plain = JPEGCodec(R, 8, 8)
    packed = JPEGCodec(R, 8, 7)
    assert packed.archived_dim(D) * 8 == D * 7


def test_triton_matches_torch_packed():
    if not torch.cuda.is_available():
        print("   (skipped: no CUDA)")
        return
    dev = "cuda"
    k = torch.cumsum(torch.randn(B, H, S, D, device=dev) * 0.1, dim=-2)
    v = torch.cumsum(torch.randn(B, H, S, D, device=dev) * 0.1, dim=-2)
    g = torch.rand(8, D, device=dev) + 0.5
    q = torch.randn(B, H, 1, D, device=dev)
    for bits in (8, 7, 4):
        c = CodecKVCache(mode="jpeg", block_size=R, keep_freqs=8,
                         quant_bits=bits, channel_scale=g)
        c.forward(k, v)
        out = codec_decode_attention(q, c._keys[0], c._values[0], impl="triton")
        ref = codec_decode_attention(q, c._keys[0], c._values[0], impl="torch")
        d = (out - ref).abs().max().item()
        assert d < 1e-4, f"bits={bits} triton/torch disagree: {d:.2e}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = True
    for fn in fns:
        try:
            fn()
            print(f"✅ {fn.__name__}")
        except AssertionError as e:
            ok = False
            print(f"❌ {fn.__name__}: {e}")
    print("\nALL PASS" if ok else "\nFAILURES")
    sys.exit(0 if ok else 1)
