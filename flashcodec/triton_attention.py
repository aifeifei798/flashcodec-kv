"""Triton kernel for coefficient-domain decode attention (split-K).

One program per (batch, kv head, query group, block-chunk) walks its slice of
the archive and performs the whole decode attention in registers:

    rows  = dequant(coeffs)                  (S, D)   per block
    s     = rows @ q                         (S,)
    logits= A @ s  * scale                   (R,)
    online-softmax update
    w     = p @ A                            (S,)
    acc  += w @ rows                         (D,)

then a tiny combine kernel merges the per-chunk (max, sum, acc) partials
(flash-decoding style) so the GPU is not limited to one program per head. That
removes the five intermediate tensors of the PyTorch path and launches two
kernels per layer instead of ~6.

Falls back to the PyTorch implementation whenever the shape/dtype/device is not
supported (see ``codec_decode_attention_triton``).
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:                                   # no triton -> torch path
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit(do_not_specialize=["nb", "nr", "chunk", "ns"])
    def _attn_split_kernel(
        Q, K, V, KS, VS, KR, VR, KG, VG, A,
        PM, PL, PA,
        nb, nr, chunk, ns,
        q_sb, q_sh, q_sq, q_sd,
        k_sb, k_sh, k_sn, k_ss, k_sd,
        v_sb, v_sh, v_sn, v_ss, v_sd,
        ks_sb, ks_sh, ks_sn, ks_ss,
        kr_sb, kr_sh, kr_st, kr_sd,
        pa_s0, pa_s1,
        SROWS: tl.constexpr, RTOK: tl.constexpr, DIM: tl.constexpr,
        BDIM: tl.constexpr, HK: tl.constexpr, GFACT: tl.constexpr,
        K_INT8: tl.constexpr, V_INT8: tl.constexpr, HAS_RAW: tl.constexpr,
        K_PACKED: tl.constexpr, V_PACKED: tl.constexpr,
        K_BITS: tl.constexpr, V_BITS: tl.constexpr,
        KPD: tl.constexpr, VPD: tl.constexpr,
        K_CHAN: tl.constexpr, V_CHAN: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        sid = tl.program_id(1)
        g = pid % GFACT
        h = (pid // GFACT) % HK
        b = pid // (GFACT * HK)

        d = tl.arange(0, BDIM)
        dm = d < DIM
        s = tl.arange(0, SROWS)
        r = tl.arange(0, RTOK)

        q = tl.load(Q + b * q_sb + (h * GFACT + g) * q_sh + d * q_sd,
                    mask=dm, other=0.0).to(tl.float32)
        a = tl.load(A + r[:, None] * SROWS + s[None, :])      # (R, S) fp32

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([BDIM], tl.float32)

        start = sid * chunk
        end = tl.minimum(start + chunk, nb)
        for n in range(start, end):
            kbase = b * k_sb + h * k_sh + n * k_sn
            if K_PACKED:                      # MSB-first bit stream, 8 values/byte
                t0 = d[None, :] * K_BITS
                j0 = t0 >> 3
                p0 = t0 & 7
                kptr = K + kbase + s[:, None] * k_ss
                b0 = tl.load(kptr + j0, mask=dm[None, :], other=0).to(tl.int32)
                b1 = tl.load(kptr + j0 + 1,
                             mask=dm[None, :] & (j0 + 1 < KPD),
                             other=0).to(tl.int32)
                u = (((b0 << 8) | b1) >> (16 - p0 - K_BITS)) & ((1 << K_BITS) - 1)
                rows = u.to(tl.float32) - (1 << (K_BITS - 1))
            else:
                rows = tl.load(K + kbase + s[:, None] * k_ss
                               + d[None, :] * k_sd,
                               mask=dm[None, :], other=0.0).to(tl.float32)
            if K_INT8:
                ksc = tl.load(KS + b * ks_sb + h * ks_sh + n * ks_sn
                              + s * ks_ss)
                rows = rows * ksc[:, None]
            if K_CHAN:                        # calibrated per-channel profile
                gk = tl.load(KG + s[:, None] * DIM + d[None, :],
                             mask=dm[None, :], other=0.0)
                rows = rows * gk
            sj = tl.sum(rows * q[None, :], axis=1)            # (S,)
            logits = tl.sum(a * sj[None, :], axis=1) * SCALE  # (R,)

            m_new = tl.maximum(m_i, tl.max(logits))
            p = tl.exp(logits - m_new)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p)

            w = tl.sum(p[:, None] * a, axis=0)                # (S,)
            vbase = b * v_sb + h * v_sh + n * v_sn
            if V_PACKED:
                t0 = d[None, :] * V_BITS
                j0 = t0 >> 3
                p0 = t0 & 7
                vptr = V + vbase + s[:, None] * v_ss
                c0 = tl.load(vptr + j0, mask=dm[None, :], other=0).to(tl.int32)
                c1 = tl.load(vptr + j0 + 1,
                             mask=dm[None, :] & (j0 + 1 < VPD),
                             other=0).to(tl.int32)
                u = (((c0 << 8) | c1) >> (16 - p0 - V_BITS)) & ((1 << V_BITS) - 1)
                vrows = u.to(tl.float32) - (1 << (V_BITS - 1))
            else:
                vrows = tl.load(V + vbase + s[:, None] * v_ss
                                + d[None, :] * v_sd,
                                mask=dm[None, :], other=0.0).to(tl.float32)
            if V_INT8:
                vsc = tl.load(VS + b * ks_sb + h * ks_sh + n * ks_sn
                              + s * ks_ss)
                vrows = vrows * vsc[:, None]
            if V_CHAN:
                gv = tl.load(VG + s[:, None] * DIM + d[None, :],
                             mask=dm[None, :], other=0.0)
                vrows = vrows * gv
            acc = acc * alpha + tl.sum(w[:, None] * vrows, axis=0)
            m_i = m_new

        # the raw tail lives in the last chunk only
        if HAS_RAW and sid == ns - 1:
            for t in range(0, nr):
                kraw = tl.load(KR + b * kr_sb + h * kr_sh + t * kr_st
                               + d * kr_sd, mask=dm, other=0.0).to(tl.float32)
                logit = tl.sum(q * kraw) * SCALE
                m_new = tl.maximum(m_i, logit)
                p = tl.exp(logit - m_new)
                alpha = tl.exp(m_i - m_new)
                l_i = l_i * alpha + p
                vraw = tl.load(VR + b * kr_sb + h * kr_sh + t * kr_st
                               + d * kr_sd, mask=dm, other=0.0).to(tl.float32)
                acc = acc * alpha + p * vraw
                m_i = m_new

        slot = pid * ns + sid
        tl.store(PM + slot, m_i)
        tl.store(PL + slot, l_i)
        tl.store(PA + slot * pa_s0 + d * pa_s1, acc, mask=dm)

    @triton.jit(do_not_specialize=["ns"])
    def _attn_combine_kernel(
        PM, PL, PA, OUT,
        ns, out_sb, out_sh, out_sq, out_sd,
        DIM: tl.constexpr, BDIM: tl.constexpr, HQT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        d = tl.arange(0, BDIM)
        dm = d < DIM
        m = float("-inf")
        l = 0.0
        acc = tl.zeros([BDIM], tl.float32)
        for i in range(0, ns):
            mi = tl.load(PM + pid * ns + i)
            li = tl.load(PL + pid * ns + i)
            ai = tl.load(PA + (pid * ns + i) * BDIM + d, mask=dm, other=0.0)
            m_new = tl.maximum(m, mi)
            alpha = tl.exp(m - m_new)
            beta = tl.exp(mi - m_new)
            l = l * alpha + li * beta
            acc = acc * alpha + ai * beta
            m = m_new
        out = acc / l
        b = pid // HQT
        hq = pid % HQT
        tl.store(OUT + b * out_sb + hq * out_sh + d * out_sd,
                 out.to(OUT.dtype.element_ty), mask=dm)


def _strides(t):
    return [t.stride(i) for i in range(t.dim())]


def codec_decode_attention_triton(query, key_stream, value_stream,
                                  scale=None, basis=None):
    """Triton implementation; returns None if the case is unsupported."""
    if not HAS_TRITON or not query.is_cuda:
        return None
    if query.shape[2] != 1:
        return None
    codec = key_stream.codec
    if codec.name != value_stream.codec.name or codec.name != "jpeg":
        return None                                    # png stores r rows: no win
    n = key_stream.n_blocks
    if n == 0:
        return None
    D = query.shape[-1]
    BD = triton.next_power_of_2(D)
    if D > 256:
        return None
    S = codec.keep_freqs
    R = key_stream.block_size
    B, Hq, _, _ = query.shape
    Hk = key_stream._coeffs.shape[1]
    if Hq % Hk != 0:
        return None
    G = Hq // Hk
    scale = D ** -0.5 if scale is None else scale

    krows, vrows = key_stream._coeffs, value_stream._coeffs
    vcodec = value_stream.codec
    int8 = codec.quantized
    ks_t, vs_t = key_stream._scales, value_stream._scales
    k_packed, v_packed = codec.packed, vcodec.packed
    kg = key_stream.codec.channel_scale
    vg = vcodec.channel_scale
    if kg is not None:
        kg = kg.to(device=query.device, dtype=torch.float32).contiguous()
    if vg is not None:
        vg = vg.to(device=query.device, dtype=torch.float32).contiguous()
    k_chan, v_chan = kg is not None and int8, vg is not None and int8
    kpd = codec.archived_dim(D) if k_packed else D
    vpd = vcodec.archived_dim(D) if v_packed else D
    if basis is None:
        from .attention import block_basis
        basis = block_basis(codec, R, device=query.device, dtype=torch.float32)
    basis = basis.contiguous()

    raw_k, raw_v = key_stream.raw, value_stream.raw
    has_raw = raw_k is not None
    nr = raw_k.shape[-2] if has_raw else 0
    dummy = krows

    n_prog = B * Hk * G
    ns = max(1, min(32, (n + 63) // 64))
    chunk = (n + ns - 1) // ns

    pm = torch.empty((n_prog, ns), dtype=torch.float32, device=query.device)
    pl = torch.empty((n_prog, ns), dtype=torch.float32, device=query.device)
    pa = torch.empty((n_prog, ns, BD), dtype=torch.float32, device=query.device)
    out = torch.empty_like(query)

    _attn_split_kernel[(n_prog, ns)](
        query, krows, vrows, ks_t if int8 else dummy,
        vs_t if int8 else dummy,
        raw_k if has_raw else dummy, raw_v if has_raw else dummy,
        kg if k_chan else dummy, vg if v_chan else dummy,
        basis, pm, pl, pa,
        n, nr, chunk, ns,
        *_strides(query),
        *_strides(krows),
        *_strides(vrows),
        *_strides(ks_t)[:4] if int8 else (0, 0, 0, 0),
        *_strides(raw_k) if has_raw else (0, 0, 0, 0),
        _strides(pa)[1], _strides(pa)[2],
        SROWS=S, RTOK=R, DIM=D, BDIM=BD, HK=Hk, GFACT=G,
        K_INT8=int8, V_INT8=int8, HAS_RAW=has_raw,
        K_PACKED=k_packed, V_PACKED=v_packed,
        K_BITS=codec.quant_bits if int8 else 8,
        V_BITS=vcodec.quant_bits if int8 else 8,
        KPD=kpd, VPD=vpd,
        K_CHAN=k_chan, V_CHAN=v_chan,
        SCALE=float(scale),
        num_warps=1,
    )
    _attn_combine_kernel[(n_prog,)](
        pm, pl, pa, out,
        ns, *_strides(out),
        DIM=D, BDIM=BD, HQT=Hk * G,
        num_warps=1,
    )
    return out
