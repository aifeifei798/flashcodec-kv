"""Materialization-free decode attention: attend directly on the compressed
archive instead of reconstructing the full KV view.

The key observation is that both codecs are **linear maps along the seq axis**:
every token inside an archived block is a fixed linear combination of the stored
rows. For a block of r tokens with stored rows ``S`` (s rows of dimension D) and
a basis matrix ``A`` (r x s),

    K_block = A @ S                      (r x D)

so the attention score of a query q against token p of that block is

    q . K[p] = (q . S^T) @ A[p]          (k dot products instead of r),

and the block's output contribution is

    sum_p prob[p] * V[p] = (prob @ A) @ S.

Both are tiny matmuls: for a single decode query the intermediates are O(S)
scores and O(s) coefficient weights — the (S, D) reconstructed KV view never
exists. jpeg uses ``A = M^T`` (r x k, the DCT-II synthesis basis) and png uses
``A = tril(ones(r, r))`` (anchor + cumulative deltas).

Scope: decoding (query length 1) against jpeg/png archives with the raw tail
handled by the standard path. Prefill still goes through the materialized view —
the coefficient-domain form would blow up the score tensor there.
"""


import torch

from .codecs import dequantize_coeffs, dct_basis


def block_basis(codec, block_size, device=None, dtype=torch.float32):
    """Synthesis matrix A (r, s) mapping the stored rows of an archived block to
    its r tokens."""
    r = block_size
    if codec.name == "jpeg":
        return dct_basis(r, codec.keep_freqs, device=device, dtype=dtype).t()
    tri = torch.tril(torch.ones(r, r, dtype=dtype, device=device))
    return tri


def _stored_rows(stream, dtype=torch.float32):
    """Stored rows of the archived blocks: (B, H, n, s, D). jpeg-int8 is
    dequantized straight into ``dtype`` (bf16 on half-precision models: halves
    the traffic and uses the tensor cores); png stays fp32 because its lossless
    promise forbids down-casting the deltas."""
    n = stream.n_blocks
    if n == 0:
        return None
    codec = stream.codec
    if codec.name == "jpeg":
        rows = stream._coeffs[:, :, :n]
        if codec.quantized:
            return dequantize_coeffs(
                rows, stream._scales[:, :, :n], dtype,
                bits=codec.quant_bits, channel_scale=codec.channel_scale,
                head_dim=stream.head_dim)
        return rows
    # png: anchor + deltas are already the stored rows; stack them on the row axis
    return torch.cat([stream._anchors[:, :, :n].unsqueeze(-2).float(),
                      stream._deltas[:, :, :n].float()], dim=-2)


def codec_decode_attention(query, key_stream, value_stream, scale=None,
                           impl="auto"):
    """Attention for one new query token against a codec archive, **without**
    materializing the reconstructed KV.

    Args:
        query: (B, Hq, 1, D) — the query for the current step.
        key_stream, value_stream: ``_CodecStream`` instances of one layer.
        scale: attention scale (default ``D ** -0.5``).
        impl: ``"auto"`` (Triton when available and supported, else torch),
            ``"torch"`` or ``"triton"``.
    Returns:
        (B, Hq, 1, D) attention output, in the query dtype.

    GQA is handled by grouping the query heads (no per-head copy of the stored
    rows). Memory scales with scores + stored rows, never with ``S * D``
    reconstructed KV — for jpeg with k < r that is r/k times smaller than raw.
    png stores r rows per block (anchor + deltas), so it is numerically correct
    here but gains nothing in memory; the coefficient path is a jpeg win.
    """
    B, Hq, Sq, D = query.shape
    if Sq != 1:
        raise ValueError("codec_decode_attention only supports a single query "
                         f"token (got Sq={Sq}); prefill should use the "
                         "materialized view")
    scale = D ** -0.5 if scale is None else scale

    if impl in ("auto", "triton") and key_stream.codec.name == "jpeg":
        from .triton_attention import codec_decode_attention_triton
        out = codec_decode_attention_triton(query, key_stream, value_stream,
                                            scale=scale)
        if out is not None:
            return out
        if impl == "triton":
            raise RuntimeError("triton implementation does not support this "
                               "case (see codec_decode_attention_triton)")

    krows = _stored_rows(key_stream, query.dtype)
    vrows = _stored_rows(value_stream, query.dtype)
    if krows is None and key_stream.raw is None:          # empty cache
        return torch.zeros_like(query)
    Hk = krows.shape[1] if krows is not None else key_stream.raw.shape[1]
    assert Hq % Hk == 0, \
        f"query heads ({Hq}) must be a multiple of kv heads ({Hk})"
    G = Hq // Hk
    wd = krows.dtype if krows is not None else torch.float32
    qg = query.to(wd).reshape(B, Hk, G, 1, D)

    logits, out = [], torch.zeros(B, Hk, G, D, dtype=wd, device=qg.device)

    if krows is not None:
        A = block_basis(key_stream.codec, key_stream.block_size,
                        device=qg.device, dtype=wd)
        s = torch.einsum("bhgqd,bhnsd->bhgns", qg, krows)   # (B,Hk,G,n,s)
        logits.append(torch.einsum("bhgns,rs->bhgnr", s, A).flatten(-2))

    if key_stream.raw is not None:                       # raw tail (full fidelity)
        logits.append(torch.einsum("bhgqd,bhtd->bhgt", qg,
                                   key_stream.raw.to(wd)))

    if not logits:                                       # empty cache
        return torch.zeros_like(query)

    probs = torch.softmax(torch.cat(logits, dim=-1).float() * scale,
                          dim=-1).to(wd)
    pos = 0
    if krows is not None:
        n, r = krows.shape[2], key_stream.block_size
        p_blk = probs[..., :n * r].reshape(B, Hk, G, n, r)
        w = torch.einsum("bhgnr,rs->bhgns", p_blk, A)    # (B,Hk,G,n,s)
        out = out + torch.einsum("bhgns,bhnsd->bhgd", w, vrows)
        pos = n * r
    if key_stream.raw is not None:
        p_raw = probs[..., pos:]
        out = out + torch.einsum("bhgt,bhtd->bhgd", p_raw,
                                 value_stream.raw.to(wd))
    return out.reshape(B, Hq, 1, D).to(query.dtype)
