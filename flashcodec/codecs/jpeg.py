"""JPEG-style lossy frequency-domain compression: blockwise orthonormal 1D
DCT-II.

Physical intuition: the hidden vectors of adjacent tokens form a strongly
autocorrelated time-series signal whose energy concentrates in low
frequencies. Each block is projected onto the first K orthonormal DCT-II basis
vectors (DC mean + AC linear trend, ...), filtering out high-frequency random
noise and yielding a geometric r/K compression ratio. Basis rows are
orthonormal, so keep=r is lossless per block.

Tensor contract: input (..., S, D); the compression/filtering axis is the
second-to-last dimension (seq axis, dim=-2). Everything is batched matmul
(GEMM), with no explicit Python loops.

Quantization
------------
``quant_bits=8`` stores coefficients as int8 with one fp32 scale per
(block, frequency) row.  ``quant_bits < 8`` additionally **packs** the values
into a contiguous MSB-first bit stream (``D * bits / 8`` bytes per row, i.e.
7 bits -> 12.5% fewer bytes) — the packed form is what the cache stores, and
both the torch and the Triton decode paths unpack it on the fly.

``channel_scale`` is a static ``(k, D)`` table (calibrated once per layer and
stream) folded into the quantizer: ``step = scale[block, freq] *
channel_scale[freq, d]``.  A single scale per frequency row is shared by all
``D`` channels, but per-channel magnitudes differ by orders of magnitude, so a
calibrated channel profile buys a lower reconstruction error at the same bit
depth (or the same error one bit narrower).  It is a constant table, so the
decode path is one extra broadcast multiply.
"""

import math

import torch
import torch.nn.functional as F

_BASIS_CACHE = {}


def dct_basis(r, k, device=None, dtype=None):
    """Precompute the orthonormal DCT-II basis M_enc (k, r):
    M[j, n] = c_j · cos(π/r · (n+½)·j), c_0 = sqrt(1/r), c_j = sqrt(2/r).
    Rows are orthonormal (M @ M.T = I); k == r gives the complete orthonormal
    basis (lossless reconstruction).
    Module-level constant cache, keyed by (r, k)."""
    key = (r, k)
    if key not in _BASIS_CACHE:
        n = torch.arange(r, dtype=torch.float32)
        j = torch.arange(k, dtype=torch.float32)
        c = torch.where(j == 0, math.sqrt(1.0 / r), math.sqrt(2.0 / r))
        m = c[:, None] * torch.cos(math.pi / r * (n[None, :] + 0.5) * j[:, None])
        _BASIS_CACHE[key] = m
    m = _BASIS_CACHE[key]
    if device is not None or dtype is not None:
        m = m.to(device=device, dtype=dtype or m.dtype)
    return m


def _pad_to_blocks(x, r):
    """Zero-pad the seq axis (-2) to a multiple of r; returns (block tensor,
    original length)."""
    S = x.shape[-2]
    n = (S + r - 1) // r
    pad = n * r - S
    if pad:
        x = F.pad(x, (0, 0, 0, pad))          # (..., S, D) pads the -2 axis only
    blocks = x.reshape(*x.shape[:-2], n, r, x.shape[-1])
    return blocks, S


def dct_compress(x, r, k):
    """Blockwise DCT projection: x (..., S, D) -> coeffs (..., n_blocks, k, D)
    (x dtype).
    The seq axis is split into blocks of r (zero-padded at the tail); each
    block is projected along the within-block axis onto the first k DCT-II
    basis vectors (computed in fp32, then cast back to the input dtype — lossy
    modes need no higher precision, and the storage ratio stays honest).
    Compression storage ratio = k/r (same dtype, plus tail padding overhead)."""
    blocks, _ = _pad_to_blocks(x, r)
    basis = dct_basis(r, k, device=x.device, dtype=torch.float32)
    hp = blocks.float().transpose(-2, -1)              # (..., n, D, r)
    coef = torch.matmul(hp, basis.t())                 # (..., n, D, k)
    return coef.transpose(-2, -1).to(x.dtype)          # (..., n, k, D)


def dct_decompress(coeffs, r, out_len=None):
    """Blockwise DCT reconstruction: coeffs (..., n_blocks, k, D) ->
    x (..., n_blocks*r, D).
    Low-frequency coefficients are multiplied by the basis to reconstruct the
    within-block waveform; when out_len is given, the tail padding is trimmed."""
    n, k = coeffs.shape[-3], coeffs.shape[-2]
    basis = dct_basis(r, k, device=coeffs.device, dtype=torch.float32)
    hp = torch.matmul(coeffs.transpose(-2, -1).float(), basis)   # (...,n,D,r)
    rec = hp.transpose(-2, -1).reshape(
        *coeffs.shape[:-3], n * r, coeffs.shape[-1])
    if out_len is not None:
        rec = rec[..., :out_len, :]
    return rec


# ---------------------------------------------------------------- quantization

def _levels(bits):
    if bits not in (2, 3, 4, 5, 6, 7, 8):
        raise ValueError(f"quant_bits must be in 2..8, got {bits}")
    return 2 ** (bits - 1) - 1


def quantize_coeffs(coeffs, bits=8, channel_scale=None):
    """Symmetric per-(block, frequency) quantization of DCT coefficients.

    coeffs (..., n, k, D) -> (q int8, scale fp32 of shape (..., n, k, 1)).
    One scale per block **and** frequency row: the DC row and the AC rows have
    very different magnitudes, and a row-wise scale costs only 4 bytes per row
    (vs D bytes for the int8 row itself).

    ``channel_scale`` (k, D) folds a calibrated per-channel profile into the
    step (``step = scale * channel_scale``); the stored ``scale`` is still one
    fp32 per row, so the archive layout is unchanged.
    """
    lvl = _levels(bits)
    g = None if channel_scale is None else channel_scale.to(
        device=coeffs.device, dtype=torch.float32)
    amax = (coeffs.float() if g is None
            else coeffs.float() / g).abs().amax(dim=-1, keepdim=True)
    scale = amax / float(lvl)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    step = scale if g is None else scale * g
    q = torch.round(coeffs.float() / step).clamp_(-lvl, lvl).to(torch.int8)
    return q, scale.to(torch.float32)


def dequantize_coeffs(q, scale, dtype=torch.float32, bits=8,
                      channel_scale=None, head_dim=None):
    """int8 coefficients (or packed bit stream when ``bits < 8``) + per-row
    scale -> coefficients in ``dtype``."""
    if bits < 8:
        q = unpack_coeffs(q, bits, head_dim=head_dim)
    out = q.to(dtype) * scale.to(dtype)
    if channel_scale is not None:
        out = out * channel_scale.to(device=out.device, dtype=dtype)
    return out


# ------------------------------------------------------------- bit packing ---
# Values are stored MSB-first in a contiguous bit stream: value i occupies
# bits [i*bits, (i+1)*bits).  Packing/unpacking is pure tensor arithmetic
# (no Python loops), so it works inside the batched archive write.

def _bit_matrix(x, bits):
    """(..., n) integer values -> (..., n*bits) uint8 bits, MSB first."""
    shifts = torch.arange(bits - 1, -1, -1, device=x.device)
    return ((x.unsqueeze(-1) >> shifts) & 1).to(torch.uint8)


def pack_coeffs(q, bits):
    """int8 coefficients (..., n, D) -> packed uint8 (..., n, ceil(D*bits/8)).

    The bit stream is padded with zero bits to the next byte boundary, so any
    head_dim works (the padding is ignored again by ``unpack_coeffs``)."""
    if bits == 8:
        return q
    D = q.shape[-1]
    total = D * bits
    nb = (total + 7) // 8
    u = (q.to(torch.int16) + (1 << (bits - 1))).to(torch.uint8)
    flat = _bit_matrix(u, bits).reshape(*q.shape[:-1], total)
    flat = F.pad(flat, (0, nb * 8 - total))
    flat = flat.reshape(*q.shape[:-1], nb, 8)
    weights = torch.arange(7, -1, -1, device=q.device, dtype=torch.uint8)
    return (flat << weights).sum(-1).to(torch.uint8)


def unpack_coeffs(packed, bits, head_dim=None):
    """Packed uint8 (..., n, ceil(D*bits/8)) -> int8 (..., n, D).

    ``head_dim`` is only needed when the packed row is byte-padded (i.e. when
    D*bits is not a multiple of 8); it is then the original D."""
    if bits == 8:
        return packed
    if head_dim is None:
        head_dim = packed.shape[-1] * 8 // bits
    flat = packed.to(torch.uint8).unsqueeze(-1)
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    flat = ((flat >> shifts) & 1).reshape(*packed.shape[:-1], -1)
    flat = flat[..., :head_dim * bits]
    mat = flat.reshape(*packed.shape[:-1], head_dim, bits).to(torch.int16)
    weights = torch.arange(bits - 1, -1, -1, device=packed.device,
                           dtype=torch.int16)
    u = (mat << weights).sum(-1)
    return (u - (1 << (bits - 1))).to(torch.int8)


class JPEGCodec:
    """JPEG-style lossy frequency-domain codec (stateless, shareable across
    caches). ``quant_bits=8`` stores the DCT coefficients as int8 with a
    per-(block, frequency) scale — a real byte-level saving on top of the k/r
    geometric ratio (the scales cost 4 bytes per row). ``quant_bits < 8``
    additionally packs the coefficients into a bit stream (7 bits = 12.5%
    smaller, and the calibrated channel profile makes 7 bits as accurate as
    uncalibrated 8 bits).

    ``channel_scale``: optional (keep_freqs, head_dim) tensor, see the module
    docstring.  The table is owned by the codec, so a calibrated schedule needs
    one codec per (layer, stream) — ``FlashCodecCache.from_config(per_layer=...)``.
    """

    name = "jpeg"

    def __init__(self, block_size=8, keep_freqs=2, quant_bits=0,
                 channel_scale=None):
        assert 1 <= keep_freqs <= block_size, \
            f"keep_freqs must be within [1, {block_size}]"
        assert quant_bits in (0, 2, 3, 4, 5, 6, 7, 8), \
            f"quant_bits must be 0 or 2..8, got {quant_bits}"
        self.block_size = block_size
        self.keep_freqs = keep_freqs
        self.quant_bits = quant_bits
        self.channel_scale = None
        if channel_scale is not None:
            self.set_channel_scale(channel_scale)

    def set_channel_scale(self, channel_scale):
        g = torch.as_tensor(channel_scale, dtype=torch.float32).detach().clone()
        assert g.dim() == 2 and g.shape[0] == self.keep_freqs, \
            (f"channel_scale must be (keep_freqs={self.keep_freqs}, head_dim), "
             f"got {tuple(g.shape)}")
        self.channel_scale = g

    @property
    def quantized(self):
        return self.quant_bits > 0

    @property
    def packed(self):
        """True when the archive stores a sub-byte bit stream."""
        return 0 < self.quant_bits < 8

    def archived_dim(self, head_dim):
        """Last-dim size of the stored coefficient tensor for ``head_dim``."""
        if not self.packed:
            return head_dim
        total = head_dim * self.quant_bits
        return (total + 7) // 8

    def compress(self, x):
        return dct_compress(x, self.block_size, self.keep_freqs)

    def quantize(self, coeffs):
        """Coefficients -> (int8 or packed uint8, scale); (None, None) when
        lossless."""
        if not self.quantized:
            return None, None
        q, scale = quantize_coeffs(coeffs, self.quant_bits,
                                   channel_scale=self.channel_scale)
        return pack_coeffs(q, self.quant_bits), scale

    def decompress(self, coeffs, out_len=None, scales=None, head_dim=None):
        if self.quantized:
            assert scales is not None, "quantized codec needs the scales"
            coeffs = dequantize_coeffs(coeffs, scales, bits=self.quant_bits,
                                       channel_scale=self.channel_scale,
                                       head_dim=head_dim)
        return dct_decompress(coeffs, self.block_size, out_len)

    def storage_ratio(self):
        """Archive bytes / original bytes **assuming fp32 input** (excluding the
        raw tail). Non-quantized: k/r. Quantized: (k/r) * bits/32."""
        if not self.quantized:
            return self.keep_freqs / self.block_size
        return (self.keep_freqs / self.block_size) * (self.quant_bits / 32.0)

    def __repr__(self):
        q = f", quant_bits={self.quant_bits}" if self.quantized else ""
        c = ", channel_scale" if self.channel_scale is not None else ""
        return (f"JPEGCodec(block_size={self.block_size}, "
                f"keep_freqs={self.keep_freqs}{q}{c})")
