"""Unified codec KV cache manager.

Design:
- Each KV stream (key/value x layer) keeps internally a ``raw_buffer``
  (tail tokens that do not fill a full block, held at full fidelity) and a
  ``compressed_storage`` (compressed archive of completed blocks: jpeg keeps
  low-frequency coefficients / png keeps anchors + deltas);
- Once block_size tokens have accumulated, batched archive compression is
  triggered (a single GEMM, no loops);
- Causal safety: only "complete blocks" are archived, the current partial
  block always stays raw — future perturbations at any point of incremental
  decoding never rewrite already-archived tensors;
- Dual interface: ``forward(key_seq, value_seq)`` tiles the full sequence
  block by block; ``decode_step(new_k, new_v)`` appends a single step and
  archives automatically across block boundaries;
- Duck-typed compatible with HuggingFace ``Cache`` (update / get_seq_length /
  key_cache / value_cache), so it can be injected directly into generate.

Tensor contract: KV layout (batch, num_heads, seq_len, head_dim), compression
axis dim=-2.
"""

import torch

from .codecs import JPEGCodec, PNGCodec, dct_decompress, png_reconstruct

try:  # native transformers 5.x Cache ecosystem (layer container mode)
    from functools import partial

    from transformers.cache_utils import Cache as _HFCache
    from transformers.cache_utils import DynamicLayer as _HFDynamicLayer
    HAS_HF_CACHE = True
except Exception:                          # fallback when transformers is absent
    _HFCache = object
    _HFDynamicLayer = object
    HAS_HF_CACHE = False


def make_codec(mode="jpeg", block_size=8, keep_freqs=2, keep_ratio=1.0,
               quant_bits=0, channel_scale=None):
    """Build a codec instance from a mode name.

    ``quant_bits=8`` switches the jpeg archive to int8 coefficients with a
    per-(block, frequency) scale (real byte-level saving); ``quant_bits < 8``
    packs the coefficients into a bit stream. ``channel_scale`` is the
    optional calibrated (keep_freqs, head_dim) profile. png ignores both (its
    lossless tier stores fp32 deltas on purpose)."""
    if mode == "jpeg":
        return JPEGCodec(block_size=block_size, keep_freqs=keep_freqs,
                         quant_bits=quant_bits, channel_scale=channel_scale)
    if mode == "png":
        if quant_bits:
            raise ValueError("quant_bits is only supported for mode='jpeg' "
                             "(the png lossless tier must keep fp32 deltas)")
        if channel_scale is not None:
            raise ValueError("channel_scale is only supported for mode='jpeg'")
        return PNGCodec(block_size=block_size, keep_ratio=keep_ratio)
    raise ValueError(f"unknown mode: {mode!r} (expected 'jpeg' | 'png')")


class _CodecStream:
    """Single (B,H,S,D) stream: raw tail + capacity-doubling preallocated archive.

    Archive storage uses std::vector-style geometric capacity doubling
    (starting at 16 blocks, doubling when full): each full block is written
    once as an in-place slice (no reallocation), and read() runs a single
    large GEMM / cumsum over the valid region — eliminating per-archive
    Python loops and fragmented kernel launches entirely.
    jpeg: ``_coeffs (B,H,capacity,k,D)`` (input dtype), or int8 +
    ``_scales (B,H,capacity,k,1)`` fp32 when the codec is quantized;
    png:  ``_deltas (B,H,capacity,r-1,D)`` (fp32) + ``_anchors (B,H,capacity,D)``.
    """

    _INITIAL_BLOCKS = 16

    def __init__(self, codec):
        self.codec = codec
        self.raw = None          # (B,H,t,D), t < block_size
        self.n_archived = 0      # total number of archived tokens
        self.n_blocks = 0        # blocks used in the archive region
        self.meta = None         # (B, H, D, elem_size)
        self.head_dim = None     # true D (the stored last dim is packed when bits<8)
        self.dtype = None        # stream input dtype (restored; SDPA dtype contract)
        self._capacity = 0       # blocks allocated in the archive (geometric growth)
        self._coeffs = None      # jpeg: (B,H,capacity,k,D)
        self._scales = None      # jpeg+int8: (B,H,capacity,k,1) fp32
        self._deltas = None      # png:  (B,H,capacity,r-1,D) fp32
        self._anchors = None     # png:  (B,H,capacity,D)
        # Incrementally decoded archive (append-only, one decode per new
        # block instead of re-decoding the whole history on every read).
        self._rec = None         # (B,H,_rec_capacity,D) input dtype
        self._rec_blocks = 0     # blocks already materialized into _rec
        self._rec_capacity = 0   # tokens allocated in _rec

    @property
    def block_size(self):
        return self.codec.block_size

    def _archive_spec(self, cap, B, H, D):
        """(attr, shape, dtype) for every archive tensor at capacity `cap`."""
        if self.codec.name == "jpeg":
            k = self.codec.keep_freqs
            if self.codec.quantized:
                packed = self.codec.packed
                dim = self.codec.archived_dim(D) if packed else D
                return [("_coeffs", (B, H, cap, k, dim),
                         torch.uint8 if packed else torch.int8),
                        ("_scales", (B, H, cap, k, 1), torch.float32)]
            return [("_coeffs", (B, H, cap, k, D), self.dtype)]
        r = self.block_size
        return [("_deltas", (B, H, cap, r - 1, D), torch.float32),
                ("_anchors", (B, H, cap, D), self.dtype)]

    def _grow(self, need_blocks, like):
        """Geometric growth: when capacity is insufficient, double until
        >= need_blocks and copy the old data over verbatim."""
        if need_blocks <= self._capacity:
            return
        cap = max(self._INITIAL_BLOCKS, 1)
        while cap < need_blocks:
            cap *= 2
        B, H, D = like.shape[0], like.shape[1], like.shape[-1]
        for attr, shape, dtype in self._archive_spec(cap, B, H, D):
            new = torch.empty(shape, dtype=dtype, device=like.device)
            if self.n_blocks:
                new[:, :, :self.n_blocks].copy_(
                    getattr(self, attr)[:, :, :self.n_blocks])
            setattr(self, attr, new)
        self._capacity = cap

    def _archive(self, head):
        """Compress full blocks head (..., n_full*r, D) into the archive
        region (in-place slice)."""
        n_full = head.shape[-2] // self.block_size
        self._grow(self.n_blocks + n_full, head)
        sl = slice(self.n_blocks, self.n_blocks + n_full)
        if self.codec.name == "jpeg":
            coeffs = self.codec.compress(head)
            q, s = self.codec.quantize(coeffs)
            if self.codec.quantized:
                self._coeffs[:, :, sl].copy_(q)
                self._scales[:, :, sl].copy_(s)
            else:
                self._coeffs[:, :, sl].copy_(coeffs)
        else:
            delta, anchor = self.codec.filter(head)
            self._deltas[:, :, sl].copy_(delta)
            self._anchors[:, :, sl].copy_(anchor)
        self.n_blocks += n_full

    def append(self, tokens):
        """Append (B,H,m,D); full blocks are batch-compressed into the archive
        immediately (a single batched GEMM)."""
        r = self.block_size
        if self.meta is None:
            self.meta = (tokens.shape[0], tokens.shape[1], tokens.shape[-1],
                         tokens.element_size())
            self.head_dim = tokens.shape[-1]
            self.dtype = tokens.dtype        # archive/read restore the input dtype
        buf = tokens if self.raw is None else torch.cat(
            [self.raw, tokens], dim=-2)
        total = buf.shape[-2]
        n_full = total // r
        if n_full > 0:
            self._archive(buf[..., :n_full * r, :])
            self.n_archived += n_full * r
        self.raw = (buf[..., n_full * r:, :].contiguous()
                    if total > n_full * r else None)

    def _decode_pending(self):
        """Materialize archive blocks that were added since the last read.

        The archive is append-only (causal-safety invariant), so the decoded
        prefix never changes: decoding only the new blocks makes `read()` O(new
        blocks) amortized instead of O(history) on every decode step."""
        if self._rec_blocks >= self.n_blocks:
            return
        r = self.block_size
        a, b = self._rec_blocks, self.n_blocks
        like = self._coeffs if self.codec.name == "jpeg" else self._deltas
        if self._rec is None or self._rec.shape[-2] < b * r:
            cap = max(b * r, self._rec_capacity * 2,
                      self._INITIAL_BLOCKS * r)
            new = torch.empty((like.shape[0], like.shape[1], cap,
                               self.head_dim),
                              dtype=self.dtype, device=like.device)
            if self._rec is not None and self._rec_blocks:
                done = self._rec_blocks * r
                new[:, :, :done].copy_(self._rec[:, :, :done])
            self._rec, self._rec_capacity = new, cap
        if self.codec.name == "jpeg":
            rec = self.codec.decompress(
                self._coeffs[:, :, a:b],
                scales=(self._scales[:, :, a:b]
                        if self.codec.quantized else None),
                head_dim=self.head_dim)
            if rec.dtype != self.dtype:
                rec = rec.to(self.dtype)
        else:
            rec = png_reconstruct(self._deltas[:, :, a:b],
                                  self._anchors[:, :, a:b])
        self._rec[:, :, a * r:b * r].copy_(rec)
        self._rec_blocks = b

    def discard_view(self):
        """Drop the memoized reconstruction buffer (it is rebuilt on demand by
        `read()`). Used by the materialization-free decode path: once prefill is
        over the view is dead weight, and decode attends on the stored rows."""
        self._rec = None
        self._rec_blocks = 0
        self._rec_capacity = 0

    def read(self):
        """Reconstructed view (B,H,S,D): one decode of the newly archived
        blocks (memoized, append-only) + concatenation of the raw tail.
        Everything is restored to the stream input dtype (attention q/k/v dtype
        contract).

        When the raw tail is empty the result is a view into the internal
        reconstruction buffer; appends only ever write past `seq_len`, and
        structural ops (crop / beam gather) never write into the live region
        in place, so a tensor handed out earlier stays valid (until an
        explicit `reset()`)."""
        if self.n_blocks == 0:
            return self.raw
        self._decode_pending()
        rec = self._rec[:, :, :self.n_archived]
        if self.raw is not None:
            rec = torch.cat([rec, self.raw], dim=-2)
        return rec

    def reset(self):
        """Zero the contents but keep the capacity: crop / refill reuse the
        allocated VRAM instead of re-allocating it over and over."""
        self.raw = None
        self.n_archived = 0
        self.n_blocks = 0
        self._rec_blocks = 0

    def crop(self, tokens_to_remove):
        """Remove trailing tokens exactly (negative count, HF semantics).

        Already-archived blocks are kept verbatim; only the partial block at
        the new boundary is re-exposed as a raw tail, taking its values from
        the current reconstruction. So a crop never re-compresses the history
        and never adds loss — unlike a rebuild-from-view crop, repeated crops
        are idempotent."""
        if tokens_to_remove > 0:
            raise ValueError("crop only supports negative values "
                             "(remove trailing tokens)")
        if tokens_to_remove == 0:
            return
        r = self.block_size
        keep = max(0, self.seq_len - abs(tokens_to_remove))
        keep_blocks = keep // r
        tail = keep - keep_blocks * r
        if keep_blocks < self.n_blocks:
            self._decode_pending()          # boundary tokens come from the view
            self.n_blocks = keep_blocks
            self.n_archived = keep_blocks * r
            self._rec_blocks = min(self._rec_blocks, keep_blocks)
            self.raw = (self._rec[:, :, keep_blocks * r:keep, :].clone()
                        if tail else None)
        elif tail == 0:
            self.raw = None
        else:
            self.raw = self.raw[..., :tail, :].contiguous()

    def select_batch(self, idx):
        """Gather rows of the batch dim (beam-search reorder / batch ops)."""
        if self.raw is None and self.n_blocks == 0 and self._rec is None:
            return
        like = (self.raw if self.raw is not None else
                (self._coeffs if self.codec.name == "jpeg" else self._deltas))
        idx = idx.to(like.device)
        r = self.block_size
        if idx.numel() == like.shape[0]:        # same batch size: in-place
            if self.raw is not None:
                self.raw = self.raw.index_select(0, idx)
            if self.n_blocks:
                for t in self._archive_tensors():
                    t[:, :, :self.n_blocks] = t[:, :, :self.n_blocks].index_select(
                        0, idx)
            if self._rec is not None and self._rec_blocks:
                done = self._rec_blocks * r
                # allocate instead of writing in place: read() may have handed
                # out views into the current reconstruction buffer
                new_r = torch.empty_like(self._rec)
                new_r[:, :, :done] = self._rec[:, :, :done].index_select(
                    0, idx)
                self._rec = new_r
            return
        B, H, D = idx.numel(), like.shape[1], self.head_dim
        for attr, shape, dtype in self._archive_spec(self._capacity, B, H, D):
            old = getattr(self, attr)
            new = torch.empty(shape, dtype=dtype, device=like.device)
            if self.n_blocks:
                new[:, :, :self.n_blocks] = old[
                    :, :, :self.n_blocks].index_select(0, idx)
            setattr(self, attr, new)
        if self._rec is not None:
            new_r = torch.empty((B, H, self._rec_capacity, D),
                                dtype=self._rec.dtype, device=like.device)
            if self._rec_blocks:
                done = self._rec_blocks * r
                new_r[:, :, :done] = self._rec[:, :, :done].index_select(0, idx)
            self._rec = new_r
        if self.raw is not None:
            self.raw = self.raw.index_select(0, idx)
        if self.meta is not None:
            self.meta = (B,) + self.meta[1:]

    def _archive_tensors(self):
        """Live archive tensors (shared by batch ops / storage accounting)."""
        if self.codec.name == "jpeg":
            return ((self._coeffs, self._scales) if self.codec.quantized
                    else (self._coeffs,))
        return (self._deltas, self._anchors)

    @property
    def seq_len(self):
        return self.n_archived + (0 if self.raw is None else self.raw.shape[-2])

    def storage_bytes(self):
        """Logical storage (valid region + raw tail), excluding the idle
        capacity reserved by the doubling preallocation and the persistent
        reconstruction view (`_rec`)."""
        total = 0
        if self.n_blocks:
            for t in self._archive_tensors():
                total += (t[:, :, :self.n_blocks].numel()
                          * t.element_size())
        if self.raw is not None:
            total += self.raw.numel() * self.raw.element_size()
        return total


class CodecKVCache:
    """Codec KV cache (multi-layer, duck-typed as an HF Cache).

    Usage::

        cache = CodecKVCache(mode="jpeg", block_size=8, keep_freqs=2)
        k, v = cache.update(k_states, v_states, layer_idx=0)   # incremental
        k, v = cache.forward(k_seq, v_seq, layer_idx=0)        # full sequence
    """

    def __init__(self, mode="jpeg", block_size=8, keep_freqs=2,
                 keep_ratio=1.0, quant_bits=0, channel_scale=None,
                 channel_scale_v=None):
        self.mode = mode
        self.codec = make_codec(mode, block_size, keep_freqs, keep_ratio,
                                quant_bits=quant_bits,
                                channel_scale=channel_scale)
        self.value_codec = make_codec(mode, block_size, keep_freqs,
                                      keep_ratio, quant_bits=quant_bits,
                                      channel_scale=channel_scale_v)
        self._keys = {}
        self._values = {}

    # ---------- internals ----------
    def _stream(self, store, layer_idx):
        st = store.get(layer_idx)
        if st is None:
            st = _CodecStream(self.value_codec if store is self._values
                              else self.codec)
            store[layer_idx] = st
        return st

    # ---------- dual interface ----------
    def update(self, key_states, value_states, layer_idx=0):
        """Incremental autoregressive interface: append new (B,H,m,D) and
        archive automatically across block boundaries; returns the layer's
        reconstructed (key_view, value_view)."""
        self._stream(self._keys, layer_idx).append(key_states)
        self._stream(self._values, layer_idx).append(value_states)
        return (self._keys[layer_idx].read(),
                self._values[layer_idx].read())

    decode_step = update

    def forward(self, key_seq, value_seq, layer_idx=0):
        """Full-sequence forward interface: tile the sequence dimension block
        by block directly (reset the layer, then fill it in one shot); returns
        the reconstructed (key_view, value_view)."""
        self.reset(layer_idx)
        return self.update(key_seq, value_seq, layer_idx)

    def reset(self, layer_idx=None):
        if layer_idx is None:
            self._keys.clear()
            self._values.clear()
        else:
            self._keys.pop(layer_idx, None)
            self._values.pop(layer_idx, None)

    def crop(self, tokens_to_remove, layer_idx=None):
        """Exact trailing-token removal (negative count) for one or all layers."""
        for store in (self._keys, self._values):
            for i, st in store.items():
                if layer_idx is None or i == layer_idx:
                    st.crop(tokens_to_remove)

    def reorder_cache(self, beam_idx, layer_idx=None):
        """Beam search: gather the batch dim of one or all layers."""
        for store in (self._keys, self._values):
            for i, st in store.items():
                if layer_idx is None or i == layer_idx:
                    st.select_batch(beam_idx)

    # ---------- HF Cache duck typing ----------
    @property
    def key_cache(self):
        mx = max(self._keys, default=-1)
        return [self._keys[i].read() if i in self._keys else None
                for i in range(mx + 1)]

    @property
    def value_cache(self):
        mx = max(self._values, default=-1)
        return [self._values[i].read() if i in self._values else None
                for i in range(mx + 1)]

    def get_seq_length(self, layer_idx=0):
        st = self._keys.get(layer_idx)
        return 0 if st is None else st.seq_len

    def get_max_length(self):
        return None                     # no upper bound

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    # ---------- diagnostics ----------
    def storage_bytes(self, layer_idx=None):
        layers = ([layer_idx] if layer_idx is not None
                  else sorted(set(self._keys) | set(self._values)))
        return sum(self._keys[i].storage_bytes() for i in layers if i in self._keys
                   ) + sum(self._values[i].storage_bytes()
                           for i in layers if i in self._values)

    def stats(self, layer_idx=0):
        ks = self._keys.get(layer_idx)
        if ks is None or ks.meta is None:
            return {"tokens": 0, "archived": 0, "raw": 0,
                    "storage_bytes": 0, "raw_bytes": 0}
        B, H, D, elem = ks.meta
        tokens = ks.seq_len
        return {"tokens": tokens, "archived": ks.n_archived,
                "raw": tokens - ks.n_archived,
                "storage_bytes": self.storage_bytes(layer_idx),
                "raw_bytes": tokens * B * H * D * elem * 2}

    def __repr__(self):
        return (f"CodecKVCache(mode={self.mode!r}, codec={self.codec}, "
                f"layers={sorted(set(self._keys) | set(self._values))})")


# ============================================================
# transformers 5.x native Cache backend (layer container mode)
# ============================================================
class FlashCodecLayer(_HFDynamicLayer):
    """transformers 5.x CacheLayer: replaces DynamicLayer's cat-style growth
    with a codec stream (raw tail + archived full blocks). update returns the
    reconstructed, causally visible KV for the attention layer to consume
    directly — the write and read replacements are both contained in this
    layer, requiring zero model surgery."""

    def __init__(self, mode="jpeg", block_size=8, keep_freqs=2,
                 keep_ratio=1.0, quant_bits=0, channel_scale=None,
                 channel_scale_v=None):
        super().__init__()
        key_codec = make_codec(mode, block_size, keep_freqs, keep_ratio,
                               quant_bits=quant_bits,
                               channel_scale=channel_scale)
        value_codec = make_codec(mode, block_size, keep_freqs, keep_ratio,
                                 quant_bits=quant_bits,
                                 channel_scale=channel_scale_v)
        self.mode = mode
        self.codec = key_codec
        self.value_codec = value_codec
        self._key_stream = _CodecStream(key_codec)
        self._value_stream = _CodecStream(value_codec)

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self._key_stream.append(key_states)
        self._value_stream.append(value_states)
        self.keys = self._key_stream.read()
        self.values = self._value_stream.read()
        return self.keys, self.values

    def get_seq_length(self):
        return self._key_stream.seq_len

    def get_max_length(self):
        return -1                            # dynamic cache, no upper bound

    def crop(self, tokens_to_remove):
        """Exact trailing-token removal (speculative decoding). Archived blocks
        are kept verbatim; only the boundary partial block falls back into the
        raw tail, so cropping never re-compresses and never adds loss."""
        self._key_stream.crop(tokens_to_remove)
        self._value_stream.crop(tokens_to_remove)
        self._rebuild_views()

    def reorder_cache(self, beam_idx):
        """Beam search: gather the batch dim of both streams. The archive and
        the raw tail have to be reordered too — reordering only the
        materialized view would silently corrupt the next read."""
        self._key_stream.select_batch(beam_idx)
        self._value_stream.select_batch(beam_idx)
        self._rebuild_views()

    def batch_repeat_interleave(self, repeats):
        """Contrastive search: replicate batch rows (archives included)."""
        if not self.is_initialized:
            return
        idx = torch.arange(self._key_stream.meta[0],
                           device=self.keys.device).repeat_interleave(repeats)
        self.select_batch(idx)

    def batch_select_indices(self, indices):
        """Contrastive search: keep only the given batch rows."""
        if not self.is_initialized:
            return
        self.select_batch(indices)

    def select_batch(self, idx):
        self._key_stream.select_batch(idx)
        self._value_stream.select_batch(idx)
        self._rebuild_views()

    def _rebuild_views(self):
        """Re-materialize the attention-facing views after a structural
        change (crop / batch gather)."""
        if not self.is_initialized:
            return
        if self._key_stream.seq_len > 0:
            self.keys = self._key_stream.read()
            self.values = self._value_stream.read()
        else:
            self.keys = torch.empty(0, dtype=self.dtype, device=self.device)
            self.values = torch.empty(0, dtype=self.dtype, device=self.device)

    def reset(self):
        self._key_stream.reset()
        self._value_stream.reset()
        super().reset()

    def offload(self):
        pass                     # prototype has no layer offload (archives are off-CPU)

    def prefetch(self):
        pass

    def storage_bytes(self):
        return (self._key_stream.storage_bytes()
                + self._value_stream.storage_bytes())


class FlashCodecCache(_HFCache):
    """transformers 5.x native codec KV cache (first-class backend).

    Usage 1 (explicit)::

        cache = FlashCodecCache(mode="jpeg", block_size=8, keep_freqs=2)
        model.generate(input_ids, past_key_values=cache, use_cache=True)

    Usage 2 (automatic injection via the apply_flash_codec hot patch).
    The Cache base class lazily creates layers through
    layer_class_to_replicate and forwards update to FlashCodecLayer;
    get_seq_length / get_max_length / crop and friends come for free by
    inheritance.
    """

    def __init__(self, mode="jpeg", block_size=8, keep_freqs=2,
                 keep_ratio=1.0, quant_bits=0, channel_scale=None,
                 channel_scale_v=None):
        if not HAS_HF_CACHE:
            raise ImportError(
                "FlashCodecCache requires transformers 5.x (this project only "
                "supports the installed 5.20 Cache ecosystem); use "
                "CodecKVCache when transformers is unavailable.")
        super().__init__(
            layers=None,
            layer_class_to_replicate=partial(
                FlashCodecLayer, mode=mode, block_size=block_size,
                keep_freqs=keep_freqs, keep_ratio=keep_ratio,
                quant_bits=quant_bits, channel_scale=channel_scale,
                channel_scale_v=channel_scale_v))
        self.mode = mode
        self.codec_cfg = dict(block_size=block_size, keep_freqs=keep_freqs,
                              keep_ratio=keep_ratio, quant_bits=quant_bits,
                              channel_scale=channel_scale,
                              channel_scale_v=channel_scale_v)

    @property
    def _seen_tokens(self):
        """Counter attribute familiar from older transformers versions (unused
        internally by 5.x, kept for compatibility)."""
        return self.get_seq_length()

    @classmethod
    def from_config(cls, config, mode="jpeg", block_size=8, keep_freqs=2,
                    keep_ratio=1.0, quant_bits=0, per_layer=None,
                    channel_scale=None, channel_scale_v=None):
        """Build the mixed-layer list from the model config (Qwen3.5-style
        hybrid architectures): only full_attention layers get a codec layer,
        while the remaining layer types (linear_attention / sliding, etc.)
        keep their native dynamic layer — consistent with how
        DynamicCache(config=...) dispatches, so native interfaces such as
        linear-attention recurrent state stay unaffected.

        ``per_layer`` overrides the codec config for individual layer indices,
        e.g. ``{3: {"keep_freqs": 4}, 7: {"quant_bits": 8}}`` — this is how a
        calibrated (non-uniform) budget is applied."""
        from transformers.cache_utils import (DYNAMIC_LAYER_TYPE_MAPPING,
                                              get_layer_types_and_kwargs)
        decoder = (config.get_text_config(decoder=True)
                   if hasattr(config, "get_text_config") else config)
        try:
            layer_types, per_layer_kwargs = get_layer_types_and_kwargs(decoder)
        except Exception:                        # legacy config, no per_layer_config
            layer_types = list(getattr(decoder, "layer_types",
                                       ["full_attention"] *
                                       getattr(decoder, "num_hidden_layers", 1)))
            per_layer_kwargs = [{} for _ in layer_types]
        per_layer = per_layer or {}
        layers = []
        for idx, (ltype, lkw) in enumerate(zip(layer_types, per_layer_kwargs)):
            if ltype == "full_attention":
                cfg = dict(mode=mode, block_size=block_size,
                           keep_freqs=keep_freqs, keep_ratio=keep_ratio,
                           quant_bits=quant_bits,
                           channel_scale=channel_scale,
                           channel_scale_v=channel_scale_v)
                cfg.update(per_layer.get(idx, {}))
                layers.append(FlashCodecLayer(**cfg))
            else:
                layers.append(DYNAMIC_LAYER_TYPE_MAPPING[ltype](**lkw))
        obj = cls.__new__(cls)
        _HFCache.__init__(obj, layers=layers, layer_class_to_replicate=None)
        obj.mode = mode
        obj.codec_cfg = dict(block_size=block_size, keep_freqs=keep_freqs,
                             keep_ratio=keep_ratio, quant_bits=quant_bits,
                             channel_scale=channel_scale,
                             channel_scale_v=channel_scale_v,
                             per_layer=per_layer)
        return obj

    def codec_layer_indices(self):
        """Indices of the layers that actually carry a codec (i.e. the
        full_attention layers in hybrid architectures)."""
        return [i for i, layer in enumerate(self.layers)
                if isinstance(layer, FlashCodecLayer)]

    def to_legacy_cache(self):
        """Export to the legacy {layer_idx: (keys, values)} format
        (reconstructed full views)."""
        return {i: (layer.keys, layer.values)
                for i, layer in enumerate(self.layers)}

    @classmethod
    def from_legacy_cache(cls, legacy, **cfg):
        """Rebuild from the legacy {(layer_idx): (k, v)} format (the whole
        span goes through the codec once more)."""
        cache = cls(**cfg)
        for layer_idx in sorted(legacy):
            k, v = legacy[layer_idx]
            cache.update(k, v, layer_idx)
        return cache

    def storage_bytes(self, layer_idx=None):
        layers = (self.layers if layer_idx is None
                  else [self.layers[layer_idx]])
        return sum(layer.storage_bytes() for layer in layers
                   if isinstance(layer, FlashCodecLayer))
