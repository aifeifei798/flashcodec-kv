"""One-step hot patch: make any HuggingFace CausalLM use FlashCodecCache.

Mechanism (zero intrusion): in standard HF models, KV writes/reads are all
funnelled through the Cache interface — after the attention layer computes
key/value it calls ``cache.update(k, v, layer_idx, ...)`` and gets the
full-history KV back. Swapping the Cache object for FlashCodecCache (a native
transformers 5.x Cache subclass) therefore completes the "replacement of the
KV write and read logic" without model surgery for every attention
implementation (GQA/MQA/RoPE variants).

- `apply_flash_codec(model, ...)`: wraps `model.generate` and `model.forward`;
  when `use_cache` is requested and no `past_key_values` is passed explicitly,
  it injects the configured FlashCodecCache; it still walks model.modules() to
  find standard KV-projection attention modules (with k_proj/v_proj) for
  validation and reporting;
- `remove_flash_codec(model)`: fully restores both wrappers.
"""

import functools
import warnings

from .cache import HAS_HF_CACHE, CodecKVCache, FlashCodecCache

__all__ = ["apply_flash_codec", "remove_flash_codec", "undo_flash_codec",
           "find_attention_modules"]


def find_attention_modules(model):
    """Find standard Self-Attention / KV projection modules (with k_proj and
    v_proj)."""
    return [m for m in model.modules()
            if hasattr(m, "k_proj") and hasattr(m, "v_proj")]


def _make_factory(mode, block_size, kwargs, sink, config=None):
    codec_kw = dict(keep_freqs=kwargs.get("keep_freqs", 2),
                    keep_ratio=kwargs.get("keep_ratio", 1.0),
                    quant_bits=kwargs.get("quant_bits", 0),
                    channel_scale=kwargs.get("channel_scale"),
                    channel_scale_v=kwargs.get("channel_scale_v"))
    per_layer = kwargs.get("per_layer")
    if HAS_HF_CACHE:
        def factory():
            c = None
            if config is not None:      # hybrid architecture (e.g. Qwen3.5):
                try:                    # only full-attention layers get a codec
                    c = FlashCodecCache.from_config(        # per layer_types
                        config, mode=mode, block_size=block_size,
                        per_layer=per_layer, **codec_kw)
                except Exception:
                    c = None
            if c is None:
                c = FlashCodecCache(mode=mode, block_size=block_size,
                                    **codec_kw)
            sink.append(c)
            return c
    else:                                       # fallback without transformers
        def factory():
            c = CodecKVCache(mode=mode, block_size=block_size, **codec_kw)
            sink.append(c)
            return c
    return factory


def apply_flash_codec(model, mode="jpeg", block_size=8,
                      materialization_free=False, **kwargs):
    """One-line hot patch: inject a codec KV cache into model (native
    transformers 5.x).

    Supports mode="jpeg" (kwargs: keep_freqs, default 2; quant_bits, default 0
    or 8 for int8 coefficients) and mode="png" (kwargs: keep_ratio, default
    1.0).
    Afterwards every `model.generate(..., use_cache=True)` call that does not
    pass past_key_values explicitly uses FlashCodecCache. With
    ``materialization_free=True`` the forward of standard Llama/Qwen-style
    attention modules is patched too, so decode steps attend on the stored rows
    directly (no reconstructed KV view). Returns model;
    `remove_flash_codec(model)` restores it at any time."""
    if getattr(model, "_flashcodec", None) is not None:
        warnings.warn("model is already patched with flashcodec, "
                      "call remove_flash_codec first")
        return model

    attns = find_attention_modules(model)
    caches = []                    # lets tests/diagnostics inspect injected caches
    factory = _make_factory(mode, block_size, kwargs, caches,
                            config=getattr(model, "config", None))

    base_generate = model.generate
    base_forward = model.forward

    @functools.wraps(base_generate)
    def wrapped_generate(*args, **kw):
        use_cache = kw.get("use_cache")
        if (use_cache is None or use_cache) and kw.get("past_key_values") is None:
            kw["past_key_values"] = factory()
        return base_generate(*args, **kw)

    @functools.wraps(base_forward)
    def wrapped_forward(*args, **kw):
        if kw.get("use_cache") and kw.get("past_key_values") is None:
            kw["past_key_values"] = factory()
        return base_forward(*args, **kw)

    model.generate = wrapped_generate
    model.forward = wrapped_forward
    n_wrapped = 0
    if materialization_free:
        from .patch_attention import apply_codec_attention
        n_wrapped = apply_codec_attention(model)
    model._flashcodec = {
        "mode": mode, "caches": caches, "attention_modules": attns,
        "codec_attention_modules": n_wrapped,
        "undo": lambda: _restore(model, base_generate, base_forward)}
    if not attns:
        warnings.warn("no attention module with k_proj/v_proj found — if the "
                      "model uses a non-standard KV interface, the patch may "
                      "have no effect")
    return model


def _restore(model, base_generate, base_forward):
    from .patch_attention import remove_codec_attention
    remove_codec_attention(model)
    model.generate = base_generate
    model.forward = base_forward
    model._flashcodec = None


def remove_flash_codec(model):
    """Undo the patch applied by apply_flash_codec (restores both generate and
    forward)."""
    info = getattr(model, "_flashcodec", None)
    if info is not None:
        info["undo"]()


undo_flash_codec = remove_flash_codec          # legacy alias
