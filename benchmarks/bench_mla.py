"""Does flashcodec work on an MLA model (DeepSeek-V2-Lite)?

The KV cache of an MLA model is not per-head K/V: DeepSeek-V2/V3 store one
``kv_lora_rank`` latent (512) plus the shared RoPE key (64) per token per layer,
and rebuild the per-head keys/values with ``kv_b_proj`` inside the attention.
flashcodec compresses whatever the cache holds, so it just works — this measures
what it costs and what it buys on the real checkpoint:

  1. the cache layout (keys = latent, values = rope key) and the storage ratio,
  2. perplexity at each tier, with and without the calibrated channel profile,
  3. the decode step at 4K, compressed vs not (MLA re-expands the whole latent
     every step, which is where the time goes),
  4. generated tokens, uncompressed vs 7-bit.

Note on the size of the win: the MLA latent is already ~7x smaller than a
comparable GQA cache (576 vs 4096 elements per token per layer for 16 heads x
128 dims), and ``expand_kv`` materializes the full per-head keys/values on every
forward. flashcodec shrinks the *cache*, not that transient expansion — which is
why the percentage saved here is the same as on a dense model but the absolute
bytes are much smaller.

Usage:
    python benchmarks/bench_mla.py
Environment:
    FLASHCODEC_MODEL        model id / path (default deepseek-ai/DeepSeek-V2-Lite)
    FLASHCODEC_MLA_WINDOWS  eval windows (default 8)
    FLASHCODEC_MLA_SEQ      window length (default 512)
    FLASHCODEC_MLA_CALIB    calibration tokens (default 1024)
    FLASHCODEC_MLA_STEP_CTX context for the decode-step timing (default 4096)
    FLASHCODEC_MAX_GPU_GIB  VRAM budget for device_map (default 80% of free)

Needs `accelerate` for the multi-device load (the 16B checkpoint does not fit in
32 GB next to the CUDA context), and transformers >= 5.22 for native MLA.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from flashcodec import (FlashCodecCache, apply_flash_codec, calibrate_channel_scales,
                        remove_flash_codec)

MODEL = os.environ.get("FLASHCODEC_MODEL", "deepseek-ai/DeepSeek-V2-Lite")
N = int(os.environ.get("FLASHCODEC_MLA_WINDOWS", "8"))
SEQ = int(os.environ.get("FLASHCODEC_MLA_SEQ", "512"))
CAL = int(os.environ.get("FLASHCODEC_MLA_CALIB", "1024"))
STEP_CTX = int(os.environ.get("FLASHCODEC_MLA_STEP_CTX", "4096"))
device = "cuda" if torch.cuda.is_available() else "cpu"


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    budget = os.environ.get("FLASHCODEC_MAX_GPU_GIB")
    kwargs = dict(local_files_only=True,
                  dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    if device == "cuda" and budget is None:
        free = torch.cuda.mem_get_info()[0] / 2**30
        budget = f"{max(1, int(free * 0.8))}GiB"
    if budget and device == "cuda":
        kwargs.update(device_map="auto", max_memory={0: budget, "cpu": "24GiB"})
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs).eval()
    if not kwargs.get("device_map"):
        model = model.to(device)
    return tok, model


def main():
    torch.manual_seed(0)
    tok, model = load()
    n_layers = model.config.num_hidden_layers
    print(f"model={MODEL}  layers={n_layers}  device={device}\n", flush=True)

    # ---- cache layout -------------------------------------------------------
    probe = tok("The quick brown fox jumps over the lazy dog. " * 40,
                return_tensors="pt").input_ids[:, :64].to(device)
    with torch.no_grad():
        out = model(input_ids=probe, use_cache=True)
    layer0 = out.past_key_values.layers[0]
    k_shape, v_shape = tuple(layer0.keys.shape), tuple(layer0.values.shape)
    k_dim, v_dim = k_shape[-1], v_shape[-1]
    print(f"cache layer 0: keys {k_shape} {layer0.keys.dtype} (latent), "
          f"values {v_shape} (rope key)")
    print(f"  {k_dim + v_dim} elements/token/layer vs "
          f"{2 * model.config.num_attention_heads * model.config.qk_nope_head_dim}"
          f" for a GQA equivalent\n", flush=True)
    del out, layer0
    torch.cuda.empty_cache() if device == "cuda" else None

    # ---- corpus -------------------------------------------------------------
    order = ["README.zh-CN.md", "README.md", "AGENTS.md", "CHANGELOG.md"]
    texts = [(ROOT / n).read_text(encoding="utf-8") for n in order
             if (ROOT / n).exists()]
    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False).input_ids)
    need = N * SEQ + CAL
    if len(ids) < need:
        sys.exit(f"corpus too small: need {need} tokens, have {len(ids)}")
    windows = [torch.tensor(ids[i * SEQ:(i + 1) * SEQ]).unsqueeze(0).to(device)
               for i in range(N)]
    calib_ids = torch.tensor(ids[N * SEQ:N * SEQ + CAL]).unsqueeze(0).to(device)
    t0 = time.time()
    calib = calibrate_channel_scales(model, calib_ids, keep_freqs=8)
    print(f"corpus {len(ids)} tokens; calibrated {len(calib)} layers in "
          f"{time.time() - t0:.1f}s", flush=True)
    if calib:
        print(f"  K profile {tuple(calib[0]['channel_scale'].shape)}, "
              f"V profile {tuple(calib[0]['channel_scale_v'].shape)}\n", flush=True)

    # ---- perplexity ---------------------------------------------------------
    def eval_ppl(cfg):
        nll = ntok = store = 0
        for w in windows:
            cache = (FlashCodecCache.from_config(model.config, block_size=8, **cfg)
                     if cfg else None)
            with torch.no_grad():
                o = model(input_ids=w, labels=w, use_cache=cache is not None,
                          past_key_values=cache)
            nll += o.loss.item() * w.shape[1]
            ntok += w.shape[1]
            if cache is not None:
                store = max(store, cache.storage_bytes())
        return math.exp(nll / ntok), store

    tiers = [("baseline", None),
             ("k=8 b=8", dict(mode="jpeg", keep_freqs=8, quant_bits=8)),
             ("k=8 b=8 +profile", dict(mode="jpeg", keep_freqs=8, quant_bits=8,
                                       per_layer=calib)),
             ("k=8 b=7", dict(mode="jpeg", keep_freqs=8, quant_bits=7)),
             ("k=8 b=7 +profile", dict(mode="jpeg", keep_freqs=8, quant_bits=7,
                                       per_layer=calib)),
             ("k=8 b=6 +profile", dict(mode="jpeg", keep_freqs=8, quant_bits=6,
                                       per_layer=calib))]
    raw = SEQ * n_layers * (k_dim + v_dim) * 2          # one window, bf16
    print(f"{'tier':<18} {'ppl':>9} {'dppl':>9} {'storage':>10} {'ratio':>7}")
    base = None
    for name, cfg in tiers:
        ppl, store = eval_ppl(cfg)
        base = base if base is not None else ppl
        ratio = "-" if not store else f"{store / raw * 100:.1f}%"
        size = "-" if not store else f"{store / 1e6:.2f} MB"
        print(f"{name:<18} {ppl:>9.4f} {ppl - base:>+9.4f} {size:>10} {ratio:>7}",
              flush=True)
    print(f"\n  ratio is against the bf16 latent cache of one {SEQ}-token window "
          f"({raw / 1e6:.1f} MB)\n", flush=True)

    # ---- decode step at STEP_CTX -------------------------------------------
    if STEP_CTX and STEP_CTX >= 64:
        text = "".join(texts)[: max(4096, STEP_CTX * 8)]
        long_ids = tok(text, return_tensors="pt",
                       truncation=True, max_length=STEP_CTX).input_ids.to(device)
        print(f"decode step at {long_ids.shape[1]} tokens "
              f"({long_ids.shape[1] * n_layers * (k_dim + v_dim) * 2 / 1e6:.0f} MB "
              f"latent):")
        for name, cfg in (("raw", None),
                          ("k=8 b=7 +profile",
                           dict(mode="jpeg", keep_freqs=8, quant_bits=7,
                                per_layer=calib))):
            cache = (FlashCodecCache.from_config(model.config, block_size=8, **cfg)
                     if cfg else None)
            with torch.no_grad():
                o = model(input_ids=long_ids, use_cache=True,
                          past_key_values=cache, logits_to_keep=1)
                pkv, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
                o = model(input_ids=nxt, past_key_values=pkv, use_cache=True,
                          logits_to_keep=1)                      # warmup
                pkv, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(5):
                    o = model(input_ids=nxt, past_key_values=pkv, use_cache=True,
                              logits_to_keep=1)
                    pkv, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
                if device == "cuda":
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / 5
            print(f"  {name:<18} {dt * 1e3:6.1f} ms/step" +
                  (f"  cache {cache.storage_bytes() / 1e6:.0f} MB" if cache else ""))
            del o, pkv, cache
            torch.cuda.empty_cache() if device == "cuda" else None
        print()

    # ---- generated tokens ---------------------------------------------------
    prompt = "用一句话解释什么是 KV cache。"
    pid = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        a = model.generate(pid, max_new_tokens=24, do_sample=False,
                           min_new_tokens=24, pad_token_id=tok.eos_token_id)
    apply_flash_codec(model, mode="jpeg", block_size=8, keep_freqs=8, quant_bits=7,
                      per_layer=calib)
    with torch.no_grad():
        b = model.generate(pid, max_new_tokens=24, do_sample=False,
                           min_new_tokens=24, pad_token_id=tok.eos_token_id)
    cache = model._flashcodec["caches"][0]
    print(f"generation: identical tokens={torch.equal(a, b)}  "
          f"codec layers={len(cache.codec_layer_indices())}  "
          f"archive={cache.storage_bytes() / 1e6:.2f} MB "
          f"for {cache.get_seq_length()} tokens")
    print(f"  {tok.decode(a[0][pid.shape[1]:], skip_special_tokens=True)!r}")
    remove_flash_codec(model)


if __name__ == "__main__":
    main()
