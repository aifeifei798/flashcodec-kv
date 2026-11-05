# flashcodec

> Compressing Transformer KV Cache tensors as **high-dimensional multimedia image signals**.

[![PyPI](https://img.shields.io/pypi/v/flashcodec-kv.svg)](https://pypi.org/project/flashcodec-kv/)
[![CI](https://github.com/aifeifei798/flashcodec-kv/actions/workflows/ci.yml/badge.svg)](https://github.com/aifeifei798/flashcodec-kv/actions/workflows/ci.yml)

[English](README.md) | [简体中文](README.zh-CN.md)

Conventional LLM research treats hidden-space tensors as isolated scalar
factors. flashcodec observes that the seq axis of a `(B, H, S, D)` KV tensor
behaves like the rows/columns of a high-dimensional image — adjacent token
representations are autocorrelated, energy concentrates in low frequencies, and
first-order differences shrink in variance (how strongly depends on the model,
layer and tensor: the measured real-KV statistics and the per-layer caveat are
in the Benchmarks section). Two classic
multimedia compression theories therefore transfer directly:

| Mode | Inspired by | Transform | Nature | Storage |
|---|---|---|---|---|
| `mode="jpeg"` | JPEG | Block-wise orthonormal 1D DCT-II, keep lowest K coeffs | **Lossy, frequency domain** | `K/r` (4×–8× less memory) |
| `mode="png"` | PNG | Per-block anchor (I-Token) + first-order deltas, Top-M pruning | **Lossless / near-lossless, time domain** | Dense, size-neutral; sparsity left to entropy coding |

`quant_bits=8` (jpeg only) stores the coefficients as int8 with a
per-(block, frequency) scale — a real byte-level saving on top of `K/r`:
**6.6% of raw KV at k=2 (15×), 46.9 dB PSNR**, versus 25% / 47.2 dB in fp32.

Math (block size `r`, operating axis `dim=-2`):

- **JPEG**: $X_{\text{blk}} \in \mathbb{R}^{r \times D}$, $C = M_{\text{enc}} X_{\text{blk}}$ with
  $M_{\text{enc}} \in \mathbb{R}^{K \times r}$, $M[j,n] = c_j \cos\frac{\pi}{r}(n+\tfrac12)j$,
  orthonormal rows ($K=r$ is per-block lossless); reconstruct $\hat X = M_{\text{enc}}^\top C$.
- **PNG**: $\Delta_i = x_i - x_{i-1}$ ($i \ge 1$), $x_0$ kept at full precision as the
  anchor; rebuild $\hat x_i = x_0 + \sum_{j=1}^{i} \Delta_j$ (`cumsum`, purely
  linear and autograd-differentiable). Autocorrelation implies
  $\mathrm{Var}(\Delta)/\mathrm{Var}(X) \approx 0.03$ (**30× variance collapse**,
  measured); after Top-M pruning the residual is nearly zero and sparse.

```
                    ┌────────────────────────────────────────────┐
  (B,H,S,D) ──append──▶  raw_buffer (tail < r, full fidelity)     │
                    │        │ r tokens accumulated, batched      │
                    │        ▼ archive (single GEMM, in-place     │
                    │             slice write into)               │
                    │  compressed_storage (capacity-doubled)      │
                    │   jpeg: coeffs (B,H,capacity,K,D)           │
                    │   png:  deltas (B,H,capacity,r-1,D)         │
                    │         + anchors (B,H,capacity,D)          │
                    └───────────────┬────────────────────────────┘
                                    │ read(): one decode of active
                                    │         region + raw tail
                                    ▼
                        reconstructed KV view (B,H,S,D)
```

Archives live in a **single geometrically-grown buffer** (std::vector-style:
start at 16 blocks, double on overflow). Each full block is one in-place
slice write — no re-allocation churn — and `read()` decodes **only the
newly archived blocks** (one batched GEMM / cumsum per new block, memoized
in an append-only reconstruction buffer) instead of re-decoding the whole
history on every decode step. (Fragmented per-block archive lists made
per-token decode ~170× slower at S=4096 due to kernel-launch overhead.)

## Research note (2026-10): tested against llama.cpp KV quantization — negative result

We tried to take this from a storage prototype to a llama.cpp KV-cache
replacement: DCT along the sequence axis, instead of or on top of llama.cpp's
row-wise KV quantization (`-ctk/-ctv`). Four models (Qwen3-0.6B, Hy-MT2-1.8B,
Gemma4-E2B, plain dense Qwen3-8B), equal bytes, scored with **mean KL +
top-1 agreement** (perplexity turned out to be a calibration artifact on
finetuned models, and it disagrees with KL about which of K/V matters):

| variant | 8-bit budget | 4-bit budget |
|---|---|---|
| truncation (`k < r`) | — | 2.7× more bytes for the same damage as `q4_0`; `k=2` unusable |
| lossless basis + per-frequency int8 | 8× more damage than `q8_0` | 13% bytes, unusable |
| DCT coefficients, ggml per-32 scales | 3× lower KL on 0.6B, neutral on 1.8B/E2B/8B | 2.3× lower KL on 0.6B (still broken), 6–170% **worse** elsewhere |
| codec stacked on `q8_0` | 0.8% fewer bytes for 63% more damage | — |

At the operating point that actually matters — `q8_0` K + `q4_0` V, 40.6% of
fp16 KV — the transform is a **no-op** on Qwen3-8B (KL 0.0047 raw vs 0.0045 with
DCT on K, top-1 97.3% vs 97.2%), while spending the 4 bits on V instead of K is
worth ~10×.

**Conclusion: not implemented in llama.cpp, and not recommended.** The seq-axis
transform only wins where row quantization has already destroyed the model;
wherever quantization is acceptable it is neutral or worse. What survives is the
mechanism study — the transform redistributes quantization error into a
slowly-varying subspace (same L2 error, 3–6× less perplexity damage), and KV
damage depends on architectural exposure and on K/V asymmetry as much as on
size. Full tables, mechanism and reproduction:
[`docs/llama-cpp-kv-quantization.md`](docs/llama-cpp-kv-quantization.md).

## Causal-safety design

Only **complete blocks** are archived; the current incomplete block always
stays in `raw_buffer` at original precision. During incremental decoding, no
future perturbation can ever rewrite archived tensors (verified bitwise in
tests). On the lossless tier (`mode="png", keep_ratio=1.0`), step-by-step
decoded attention matches full-sequence attention within ≤1e-4.

## Installation

```bash
pip install flashcodec-kv            # core: torch>=2.0
pip install "flashcodec-kv[hf]"      # + transformers 5.x backend
pip install "flashcodec-kv[demo]"    # + gradio/matplotlib for the demo
pip install -e ".[hf]"               # or from source
```

Distribution name is `flashcodec-kv`; the import name stays `flashcodec`. (The
bare `flashcodec` name on PyPI belongs to an unrelated placeholder package, so
don't `pip install flashcodec`.)

## Usage

### 1. Standalone codec cache (no transformers)

```python
import torch
from flashcodec import CodecKVCache

cache = CodecKVCache(mode="jpeg", block_size=8, keep_freqs=2, quant_bits=8)
k_hat, v_hat = cache.forward(k_seq, v_seq)   # prefill: (B,H,S,D) -> reconstructed view
k1, v1 = cache.decode_step(k_new, v_new)     # one new token (alias: update)
print(cache.stats(0))        # tokens / archived / raw / storage_bytes / raw_bytes
print(cache.storage_bytes()) # archive + raw tail, all layers
```

`forward` resets that layer and refills it; `decode_step` appends and archives
whole blocks. Switch to `mode="png", keep_ratio=1.0` for the bitwise-lossless
tier. `crop(-n)` removes trailing tokens (negative count, HF convention),
`reorder_cache(beam_idx)` does the beam-search gather, `reset()` keeps the
allocated capacity.

### 2. transformers 5.x: pass the cache explicitly

```python
from flashcodec import FlashCodecCache

cache = FlashCodecCache.from_config(
    model.config, mode="jpeg", block_size=8,
    keep_freqs=8,      # k = r: full DCT basis (lossless transform)
    quant_bits=8)      # int8 coefficients -> ~52% of bf16 KV
out = model.generate(input_ids, max_new_tokens=64,
                     past_key_values=cache, use_cache=True)

print(cache.get_seq_length())        # prompt + new_tokens - 1
print(cache.storage_bytes())         # codec layers: archive + raw tail
print(cache.codec_layer_indices())   # e.g. [3, 7, 11, 15, 19, 23]
```

`from_config` attaches a codec only to `full_attention` layers, so hybrid models
(Qwen3.5-style linear + full attention) keep the linear layers native; on a
plain decoder-only model every layer is compressed. `FlashCodecCache` is a
regular `transformers.cache_utils.Cache` subclass (5.x layer API), so
`update` / `get_seq_length` / `crop` / `reorder_cache` behave like the native
cache.

### 3. One-line hot patch

```python
from flashcodec import apply_flash_codec, remove_flash_codec

apply_flash_codec(model, mode="jpeg", block_size=8,
                  keep_freqs=8, quant_bits=8)
out = model.generate(input_ids, max_new_tokens=64)   # codec cache injected
remove_flash_codec(model)                            # back to the stock cache
```

Accepted kwargs: `keep_freqs` (jpeg), `keep_ratio` (png), `quant_bits`
(`0` or `2..8`; below 8 the coefficients are bit-packed), `channel_scale` /
`channel_scale_v` (calibrated channel profiles), `per_layer={idx: {...}}` for a
per-layer budget, and `materialization_free=True` for the decode path below.

### 4. Picking a tier

| goal | config | storage (bf16 KV) | cost (measured, Qwen3-0.6B) |
|---|---|---|---|
| bitwise-identical output | `mode="png", keep_ratio=1.0` | ~1.9× | storage only (fp32 deltas) |
| **recommended (calibrated)** | `keep_freqs=8, quant_bits=7` + profile | **~45%** | +0.025 ppl |
| tighter (calibrated) | `keep_freqs=8, quant_bits=6` + profile | **~39%** | +0.249 ppl |
| safe without calibration | `keep_freqs=8, quant_bits=8` | ~52% | +0.52 ppl |
| 7-bit without calibration | `keep_freqs=8, quant_bits=7` | ~45% | +2.34 ppl |
| truncation (few codec layers) | `keep_freqs=2` / `4` | 25–50% | collapses on dense models |
| maximum | `keep_freqs=2, quant_bits=8` | ~13% | collapses on dense models |

Rule of thumb: `keep_freqs = block_size` (full DCT basis) is the premise;
**dropping bits beats truncating frequencies**, but only after the channel
profile is calibrated — below 7 bits the archive collapses without it. Hybrid
models (few codec layers) tolerate truncation much better; measured `k=4` is
even slightly better than the baseline there.

#### 4.1 Calibrated channel profile (the prerequisite for 7-bit)

Each coefficient row has **one** fp32 scale shared by all 128 channels, but
per-channel magnitudes differ by orders of magnitude, so most quantizer levels
are spent on a handful of loud channels. `calibrate_channel_scales` runs one
forward pass (1–2k tokens) and estimates a static `(keep_freqs, head_dim)` table
per layer, separately for K and V:

```python
from flashcodec import calibrate_channel_scales

ids = tokenizer(calib_text, return_tensors="pt").input_ids.to("cuda")
calib = calibrate_channel_scales(model, ids, keep_freqs=8)   # one forward
apply_flash_codec(model, mode="jpeg", keep_freqs=8, quant_bits=7,
                  per_layer=calib)
```

Same Qwen3-0.6B, 16 x 512-token windows, profile calibrated on a disjoint
2048-token slice:

| tier | ppl | dppl | storage |
|---|---|---|---|
| baseline | 31.1504 | — | 100% |
| k=8 b=8 | 31.6745 | +0.524 | 51.6% |
| k=8 b=8 + profile | 31.1210 | −0.029 | 51.6% |
| k=8 b=7 | 33.4916 | +2.341 | 45.3% |
| **k=8 b=7 + profile** | **31.1750** | **+0.025** | **45.3%** |
| k=8 b=6 | 47.2046 | +16.05 | 39.1% |
| k=8 b=6 + profile | 31.3993 | +0.249 | 39.1% |

Hybrid Qwen3.5-0.8B (only 6/24 layers carry a codec, baseline 23.5318):
b=7 + profile +0.015, b=6 + profile −0.010. Reproduce with
`python benchmarks/bench_calibrated.py` (windows are built from the repository's
own markdown; the calibration slice never overlaps the eval windows).

The profile is a codec constant, so decode pays one broadcast multiply:
at 64K / k=8 the Triton path goes 0.371 -> 0.414 ms per layer; 7-bit adds the
gather unpack on top (0.767 ms per layer). `per_layer` may cover only some
layers.

#### 4.2 Archiving: entropy coding is for disk / transport only

On the hot path the int8 coefficients are read by the attention kernel directly,
so re-decoding an entropy stream every step would lose; on disk the symbols
carry ~6.8/8 bits of entropy, so compressing is free money. `save_archive`
compresses the logical int8 stream + scales + tail (zstd when importable,
otherwise zlib):

```python
from flashcodec import save_archive, load_archive

stats = save_archive(cache, "ctx.fckv")   # {"raw_bytes","stored_bytes","ratio"}
cache2 = load_archive("ctx.fckv", config=model.config, device="cuda")
```

28-layer Qwen3-0.6B, 4001 tokens, k=8 b=7 + profile: 208.0 MB in memory ->
181.6 MB on disk (76.7% of the logical payload, 39.6% of the bf16 KV), bitwise
round trip; save 5.0 s, load 13.2 s on CPU / 1.46 s with `device="cuda"` (the
7-bit re-pack is ~100x faster on the GPU).

### 5. Materialization-free decode (no KV view)

```python
apply_flash_codec(model, mode="jpeg", block_size=8, keep_freqs=8,
                  quant_bits=8, materialization_free=True)
out = model.generate(input_ids, max_new_tokens=64)

from flashcodec import codec_attention_stats
print(codec_attention_stats(model))   # wrapped attention modules (0 = patch off)
```

Decode steps attend on the stored DCT rows with the Triton split-K kernel, and
the prefill view is released, so steady-state memory is the archive only. The
patch is guarded (single-token, batch=1, eval, no sliding window, standard
`position_embeddings`) and silently falls back otherwise.

### 6. Diagnostics

| call | returns |
|---|---|
| `cache.stats(layer)` | `tokens / archived / raw / storage_bytes / raw_bytes` |
| `cache.storage_bytes(layer=None)` | archive + raw tail bytes (excludes reserved capacity and the reconstructed view) |
| `cache.codec_layer_indices()` | layers that actually carry a codec |
| `codec_attention_stats(model)` | number of wrapped attention modules (`0` = patch off) |
| `HAS_HF_CACHE` | whether the transformers backend is importable |

### 7. Codec-level API (no cache)

```python
from flashcodec import JPEGCodec, PNGCodec

codec = JPEGCodec(block_size=8, keep_freqs=2, quant_bits=8)
coeffs = codec.compress(k_seq)              # (..., n_blocks, k, D)
q, scale = codec.quantize(coeffs)           # int8 + per-(block, freq) scale
k_hat = codec.decompress(q, scales=scale)   # (..., S, D), input dtype

png = PNGCodec(block_size=8, keep_ratio=1.0)
delta, anchor = png.filter(k_seq)
k_hat = png.reconstruct(delta, anchor)
```

Tensor contract: `(B, H, S, D)`, compression axis `dim=-2`. The functional API
(`dct_compress`, `dct_decompress`, `quantize_coeffs`, `dequantize_coeffs`,
`png_filter`, `png_reconstruct`, `dct_basis`) is exported too.

### 8. Pitfalls

- `pip install flashcodec-kv`, but `import flashcodec` — the PyPI name is the
  distribution, not the module.
- From a source checkout, `import flashcodec` must resolve to the package dir,
  not the project dir of the same name (`pip install -e .`, or run inside it).
- `crop` takes a **negative** count (`cache.crop(-4)` removes 4 trailing tokens).
- Beam search works on every tier, but only `png keep_ratio=1.0` is
  bitwise-identical to the unpatched model.
- `materialization_free=True` is decode-only and batch=1; batched/padded
  generation falls back automatically.
- `quant_bits` is jpeg-only (the png lossless tier must keep fp32 deltas).

### 9. Export a 7-bit model bundle

flashcodec compresses the **KV cache at inference time**; the weights are never
quantized (they stay bf16). "Exporting a 7-bit model" therefore means the
original checkpoint plus a `flashcodec-kv.json` sidecar carrying the codec tier
and the per-layer, per-stream (K/V) channel profiles:

```bash
python -m flashcodec.export Qwen/Qwen3-0.6B ./qwen3-0.6b-fckv \
    --calib-text calibration.txt        # add --save-weights for a self-contained dir
python -m flashcodec.export --inspect ./qwen3-0.6b-fckv

# checkpoints that do not fit on one device (needs accelerate):
python -m flashcodec.export deepseek-ai/DeepSeek-V2-Lite ./dsv2-lite-fckv \
    --calib-text calibration.txt --device-map auto   # 80% of free VRAM by default
```

```text
format   : flashcodec-kv-model v1
base     : Qwen/Qwen3-0.6B  (weights referenced)
codec    : jpeg block=8 keep_freqs=8 quant_bits=7
calib    : 2048 tokens, q0.95
layers   : 28
profiles : [(8, 128)]
storage  : ~45.3% of bf16 KV (one stream)
```

Load it (`flashcodec-export` is the same command as a console script):

```python
from flashcodec import load_pretrained

model, tok = load_pretrained("./qwen3-0.6b-fckv")
inputs = tok("Explain KV cache compression in one sentence.",
             return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128)
print(model._flashcodec["caches"][0].storage_bytes())   # actual archive
```

`load_pretrained` loads the weights, applies `apply_flash_codec` with the 7-bit
tier and the stored profiles, and generation uses it immediately. If you loaded
the model yourself, use `apply_export(model, read_manifest(path))`. To export a
different tier just change `--quant-bits` — the profiles are tier-independent
and reused.

Measured on Qwen3-0.6B with 2048 calibration tokens: 313 KB for 28 layers,
45.3% of bf16 KV, and the generated tokens are identical to the uncompressed
model on the same prompt.

**Compatibility**: any decoder-only CausalLM on transformers 5.x whose cache is a
`Cache` with `.layers`. Dense models calibrate every layer; hybrid architectures
only calibrate the `full_attention` layers (measured on Qwen3.5-0.8B: 6/24
layers, 133 KB bundle, token-identical generation); sliding-window layers keep
the native cache and get no codec. Architectures without a KV cache
(Mamba/RWKV) fail loudly. Custom architectures need `--trust-remote-code`, and
the calibration length is clamped to the model's `max_position_embeddings`.

**MLA (DeepSeek-V2/V3)**: the cache stores one latent per token per layer — the
`kv_lora_rank` compressed key plus the shared RoPE key (576 elements for
V2-Lite) — instead of per-head K/V, and flashcodec compresses exactly that
tensor, so MLA works unchanged. Measured on the real DeepSeek-V2-Lite
checkpoint (`benchmarks/bench_mla.py`, 27 layers): 7-bit + profile is 44.4% of
the bf16 latent cache, ppl 15.2497 -> 15.2369 (6-bit + profile 15.3142), and the
generated tokens are identical. Two caveats: the MLA latent is already ~7x
smaller than a GQA cache (576 vs 4096 elements/token/layer at 16 heads x 128),
and transformers' `expand_kv` rebuilds the per-head keys/values on every
forward, so only the *stored* cache shrinks — the 4K decode step measured
954 ms raw vs 959 ms with the codec. When the checkpoint does not fit on one
device, export with `--device-map auto` (needs `accelerate`; it budgets 80% of
free VRAM plus host RAM, and `--max-gpu-gib` overrides the VRAM figure).


## Transformers integration (real-world tested on Qwen3.5-0.8B)

Hybrid architectures (e.g. Qwen3.5: 18 linear-attention layers + 6
full-attention layers) are handled automatically —
`FlashCodecCache.from_config(model.config)` attaches codec layers **only to
`full_attention` layers** and keeps native dynamic layers for everything
else (recurrent/conv states untouched). `apply_flash_codec` uses `from_config`
whenever a model config is available.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from flashcodec import apply_flash_codec

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B",
                                             torch_dtype="bfloat16").cuda()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
ids = tok("A quiet lake at dawn", return_tensors="pt").input_ids.cuda()

# One line — every generate() now runs on the codec cache
apply_flash_codec(model, mode="jpeg", block_size=8, keep_freqs=2)
out = model.generate(ids, max_new_tokens=48, do_sample=False)
```

Measured on Qwen3.5-0.8B (bf16, CUDA, 48 greedy tokens, `benchmarks/run_qwen_demo.py`):

| Tier | Storage vs raw KV | Output | Verdict |
|---|---|---|---|
| `png keep=1.0` | ~1.8× (fp32 deltas for exactness) | **bit-identical tokens to the unpatched model** | 100% fidelity baseline |
| `jpeg k=8 int8` | **55.5%** (~1.8×) | fluent, **100% token match** | safe lossy tier |
| `jpeg k=2` | **32.3%** (~3.1×) | fluent, 38.1% token match | lossy tier |
| `jpeg k=4` | 54.8% (~1.8×) | fluent, 49.2% token match | lossy tier |
| `jpeg k=2 int8` | **21.1%** (~4.7×) | fluent, 38.1% token match | lossy + int8 |

Hybrid-layer audit: `codec_layers=6/24` (full attention only), zero warnings.
Reproduce with `python benchmarks/run_qwen_demo.py` (offline; honors
`FLASHCODEC_MODEL` / `FLASHCODEC_NEW_TOKENS`).

dtype contracts (see `AGENTS.md`): PNG deltas are always fp32 (bf16
differences lose the bitwise-exactness the lossless tier promises); JPEG
coefficients and `read()` views stay in the stream's input dtype (SDPA
requires q/k/v dtype parity).

## Materialization-free decode attention

Every codec here is a **linear map along the seq axis**: the r tokens of a block
are `K_block = A @ S`, where `S` are the stored rows (DCT coefficients, or
anchor + deltas) and `A` is the synthesis matrix. So a decode query can attend
on `S` directly:

```
q . K[p] = (q . S^T) @ A[p]        out = (prob @ A) @ S
```

`codec_decode_attention(q, key_stream, value_stream)` implements this for one
new token (jpeg k<r, jpeg int8 and png; GQA grouping included). The `(S, D)`
reconstruction never exists, and the result matches the materialized path to
≤1e-6.

Measured (`benchmarks/bench_attention.py`, B=1, H_kv=8, H_q=16, D=64, fp32):

| seq | mode | materialized | coefficient-domain |
|---|---|---|---|
| 16384 | jpeg k=2 int8 | 0.20ms / 130MB | 0.30ms / **32MB** |
| 65536 | jpeg k=2 int8 | 0.86ms / 520MB | **0.46ms / 128MB** |
| 65536 | png keep=1.0 | 0.87ms / 520MB | 1.03ms / 272MB |

Scope: **decode only** (prefill still uses the view — the coefficient form would
explode the score tensor). png stores r rows per block, so it is numerically
supported but gains nothing; the win is jpeg `k < r`.

A **Triton split-K kernel** (`flashcodec/triton_attention.py`, flash-decoding
style: block chunks write partial max/sum/acc, a combine kernel merges them)
removes the intermediate tensors and the launch overhead. One layer, B=1,
H_kv=8, H_q=16, D=128, `jpeg k=2 int8` (`benchmarks/bench_attention.py --cuda`):

| seq | materialized | coefficient (torch) | coefficient (Triton) |
|---|---|---|---|
| 16384 | 0.20ms / 130MB | 0.29ms / 24MB | **0.09ms / 0MB** |
| 65536 | 0.86ms / 520MB | 0.41ms / 96MB | **0.16ms / 0MB** |

`apply_flash_codec(model, ..., materialization_free=True)` wires it into
transformers: the forward of standard Llama/Qwen-style attention modules is
wrapped (projection + RoPE unchanged, the attention output replaced), guarded to
single-token / batch-1 / eval / non-sliding layers, otherwise it falls back to
the original path. The prefill reconstruction is released on the first decode
step. `remove_flash_codec` restores everything.

End-to-end (`benchmarks/bench_e2e.py`, Qwen3-0.6B, 4096-token prompt + 32 new):

| tier | tok/s | peak | steady |
|---|---|---|---|
| baseline (raw KV) | 52.1 | 581MB | 0MB |
| codec k=2 int8, materialized | 44.2 | 1545MB | 1535MB |
| codec k=2 int8, materialization-free (Triton) | 40.3 | **642MB** | **122MB** |

**12.6× less steady memory** at ~9% slower decode than the materialized path
(per decode step at 4096 tokens: 22.8ms vs 21.7ms materialized, 17.8ms raw).
The remaining gap is the per-layer split/combine launch pair; a persistent
kernel would close it.

## Demo (side-by-side chat on Qwen3.5-0.8B)

![flashcodec side-by-side Gradio demo](images/demo.png)

Same prompt, two panels: native KV cache on the left, flashcodec on the right
(five tiers: png lossless / jpeg k=8 int8 / k=4 / k=2 / k=2 int8), with a
materialization-free toggle (coefficient-domain + Triton) and per-turn stats:
KV footprint (archive vs materialized view), peak/steady memory, per-token
latency and token match.

The **Full comparison** tab runs the baseline plus every tier on one prompt and
reports, per tier: KV footprint vs raw, archive/view split, decode tok/s,
peak/steady memory, teacher-forced NLL of the continuation (a perplexity-style
quality number) and token match — plus storage / speed / NLL bars.

```bash
pip install -e ".[demo]"   # gradio + matplotlib
python demo/qwen_gradio_chat.py    # Gradio side-by-side + savings chart
```

## Benchmarks (measured; B=2 H=8 S=2048 D=64, random-walk autocorrelated signal, fp32)

### Reconstruction quality (`benchmarks/bench_reconstruct.py`)

| Config | PSNR (dB) | cosine | Storage |
|---|---|---|---|
| jpeg r=8 k=1 (8×) | 42.95 | 0.99943 | 12.5% |
| jpeg r=8 k=2 (4×) | 47.22 | 0.99985 | 25.0% |
| jpeg r=8 k=4 (2×) | 51.66 | 1.00001 | 50.0% |
| jpeg r=8 k=2 int8 | 46.92 | 0.99983 | 6.6% |
| jpeg r=8 k=4 int8 | 50.89 | 0.99999 | 13.3% |
| png r=8 keep=0.5 | 50.11 | 0.99997 | 100% (sparsity → entropy coder) |
| png r=8 keep=1.0 | ~196 (fp rounding) | 1.00000 | 100% |

### Real KV statistics (real models, 512 tokens)

`bench_reconstruct` uses a synthetic random walk; real caches are messier.
Measured on real post-RoPE KV (`Var(Δ)/Var(X)` along the seq axis, and the
fraction of block energy kept by the lowest k=2 DCT coefficients of a block of
8):

| model / layer | tensor | Var(Δ)/Var(X) | energy in k=2 |
|---|---|---|---|
| Qwen3-0.6B L0 | K | 0.16 | 0.94 |
| Qwen3-0.6B L0 | V | 1.87 | 0.31 |
| Qwen3-0.6B L27 | V | 1.02 | 0.59 |
| Qwen3.5-0.8B L3 | K | 0.72 | 0.71 |
| Qwen3.5-0.8B L23 | V | 0.09 | 0.97 |

Values are frequently **anti**-correlated (ratio > 1), so "first differences
collapse the variance" is a property of smooth synthetic signals, not of every
real layer. Practical consequence: coefficient truncation (`k < r`) is a
per-layer decision — on a dense model (28 compressed layers) the error compounds
and quality collapses; use `k = r` + `quant_bits=8` there.

### Perplexity (`benchmarks/bench_ppl.py`; 16 × 512 tokens, bf16, local corpus)

| tier | Qwen3.5-0.8B (6/24 layers) | Qwen3-0.6B (28/28 layers) | storage |
|---|---|---|---|
| baseline | 22.4803 | 16.7931 | 100% |
| png keep=1.0 | 22.4803 (+0.000) | 16.7931 (+0.000) | 188% |
| jpeg k=8 (lossless basis) | 22.4877 (+0.007) | 16.8058 (+0.013) | 100% |
| **jpeg k=8 int8** | **22.4991 (+0.019)** | **17.1348 (+0.34)** | **~52%** |
| jpeg k=4 | 22.3010 (−0.18) | 470.85 (+454) | 50% |
| jpeg k=2 | 24.0566 (+1.58) | 6485.5 (+6469) | 25% |
| jpeg k=2 int8 | 24.0590 (+1.58) | 8118.5 (+8102) | 13% |

The hybrid result is good because only 6 of 24 layers carry a codec (the 18
linear-attention layers keep their native state); the dense model shows why
per-layer truncation must not be applied blindly.

### Per-layer budgets (`benchmarks/bench_adaptive.py`, Qwen3-0.6B, 6 × 512 tokens)

Calibration: compress one layer at a time to k=4, measure the perplexity damage,
then give the cheapest half of the layers k=4 and leave the rest at k=r (75%
storage):

| schedule | ppl | Δppl | storage |
|---|---|---|---|
| baseline (k=r) | 18.9786 | — | 100% |
| uniform k=6 | 48.0531 | +29.07 | 75% |
| adaptive (14 layers k=4) | 40.8309 | +21.85 | 75% |
| uniform k=4 | 710.22 | +691.24 | 50% |
| uniform k=r int8 | 19.3627 | +0.38 | 50% |

Calibration beats a uniform budget at the same storage (−15%), but the damages
are **super-additive** (the cheapest 14 layers add up to +22 ppl although no
single one costs more than +0.5), and int8 at 50% is far better. Use
`per_layer={idx: {"keep_freqs": ...}}` when a budget forces structural
truncation; otherwise prefer `quant_bits=8`.

### Cache footprint & latency (`benchmarks/bench_vram.py`, CPU)

| seq | Mode | storage | ratio | fill | read |
|---|---|---|---|---|---|
| 4096 | jpeg k=2 | 4.0MB | 0.25 | 6.7ms | 0.0ms |
| 16384 | jpeg k=2 | 16.0MB | 0.25 | 34.3ms | 0.5ms |
| 65536 | jpeg k=2 | 64.0MB | 0.25 | 141.9ms | 3.1ms |
| 4096 | jpeg k=2 int8 | 1.1MB | 0.07 | 9.2ms | 0.0ms |
| 16384 | jpeg k=2 int8 | 4.3MB | 0.07 | 46.0ms | 0.3ms |
| 65536 | jpeg k=2 int8 | 17.0MB | 0.07 | 197.9ms | 3.1ms |
| 65536 | png keep=1.0 | 256.0MB | 1.00 | 243.3ms | 3.3ms |

CPU timings vary ~±20% between runs (min of 3); the storage/ratio columns are
exact. `read` = materialize the view and touch it (`.sum()`, so the number is the
consumer cost, not just kernel launch). Decoding is memoized per archived block,
so `read` costs one decode per **new** block instead of re-decoding the whole
history on every decode step (previously 38.7ms / 71.9ms in the last two rows).

PNG does not shrink dense storage (the fate of a lossless tier) — its value is
the 30× variance collapse plus near-zero sparse residuals after Top-M, i.e.
ideal input for an entropy coder (none is bundled).

## Same-bytes comparison with llama.cpp KV quantization

llama.cpp ships row-wise KV quantization (`-ctk/-ctv`), so the fair question is
not "does the codec compress" but "does it beat what llama.cpp already has at
the same byte budget". Measured on the same model (Qwen3-0.6B, wikitext,
llama.cpp `184f65360`, CUDA + FA):

| KV config | ppl | Δ vs fp16 | of fp16 bytes |
|---|---|---|---|
| `-ctk f16 -ctv f16` | 19.6037 | — | 100% |
| `-ctk q8_0 -ctv q8_0` | 19.6138 | +0.05% | 53.1% |
| `-ctk q8_0 -ctv q4_0` | 19.6579 | +0.28% | 40.6% |
| `-ctk q4_0 -ctv q4_0` | 65.8321 | +236% | 28.1% |

Against that bar, flashcodec's **current archive format** does not win at equal
bytes:

- **truncation (`k < r`)**: `jpeg k=4` on K (75% of fp16 K bytes) damages as
  much as `q4_0` on everything (28% of fp16 KV bytes) — 2.7× more bytes for the
  same damage; `k=2` is unusable on dense models.
- **lossless basis + `quant_bits=8`** (≈50% of fp16 bytes): +0.38% ppl at
  b32/k32, versus +0.05% for `q8_0` — 8× more damage for the same budget.
- **stacking on top of llama.cpp's quantizer** does not help either: on `q8_0`
  the codec reclaims 0.8% of the bytes (one scale per 128 values instead of one
  per 32) for 63% more damage, and on `q4_0` an int8 coefficient archive is
  *larger* than the 4-bit payload it replaces (39.8% vs 28.1%).

**The coefficient quantizer is what loses, not the transform.** Give the
coefficients the same per-32 absmax that ggml uses (lossless basis, identical
byte count) and the ordering reverses: **5× less perplexity damage at 8 bits and
7× less at 4 bits**, with the reconstruction error unchanged — the transform
shifts the quantization error into the slowly-varying subspace, which softmax
barely sees (error shaping, not a better K). Measured in HF; still to be
confirmed inside llama.cpp, and it does not make 4-bit K usable on this model
(ppl 41.6 vs 18.6).

The measurements also show K and V are asymmetric: 4-bit **K** is the only
cliff (K alone at `q4_0` → +237%), while 4-bit V is nearly free (+0.29%). So
the cheap production setting is `-ctk q8_0 -ctv q4_0`. But that cliff is a
**small-model** property: on a 27B hybrid (`qwen35`, 17 KV-bearing layers)
`q4_0` KV costs +0.39% at 8k context and +0.20% at 32k, so production-scale
models have no byte-budget problem for this codec to solve.

**Dense mid-size models are where 4-bit KV is costly but not fatal**, and also
where perplexity stops working as a metric. Gemma4-12B (48 layers, 40 SWA + 8
global MQA) measured with `--kl-divergence`: KL 0.17 at 8 bits, 0.30 at 4 bits,
top-1 agreement 89% (versus 0.0035 on the 27B hybrid). Its ppl ranges from 2.2
to 355 depending on the corpus and sometimes *improves* under quantization, so
ppl cannot be used there.

**The HF-side KL pre-judgment then closes the follow-up**: at equal bytes the
transform helps only where row quantization is already destructive (Qwen3-0.6B:
3× lower KL at 8 bits, 2.3× at 4 bits but still broken), is neutral at 8 bits /
slightly worse at 4 bits on Hy-MT2-1.8B and Gemma4-E2B, and on a **plain dense
Qwen3-8B** — the most favourable case — it changes nothing at the operating
point that matters (`q8_0` K + `q4_0` V at 40.6% bytes: KL 0.0047 raw vs 0.0045
with DCT on K, top-1 97.3% vs 97.2%) while being 6% worse at 4 bits on both
sides. There is no operating point where it is worth shipping — so **no
llama.cpp integration follows from this**. Full tables, mechanism and
reproduction:
[`docs/llama-cpp-kv-quantization.md`](docs/llama-cpp-kv-quantization.md).

## Why not offload V to host / a video stream (measured)

The proposal: keep K in VRAM (high-frequency, exact matching), pack V into a
JPEG/AV1 stream on host RAM or disk, and decode it with NVDEC asynchronously —
breaking the HBM wall. Three measurements close that path
(`python benchmarks/bench_offload.py`; Qwen3-0.6B on an RTX 5090 D, PCIe Gen4
x8, pinned H2D measured at 10.7 GB/s):

| context | decode step | V (8-bit) | H2D | xfer/step | concurrent step | NVDEC/step |
|---|---|---|---|---|---|---|---|
| 4K | 16.1 ms | 117 MB | 11.0 ms | 0.7x | 18.4 ms (1.1x) | 37.4 ms |
| 16K | 16.1 ms | 470 MB | 44.0 ms | 2.7x | 47.3 ms (2.9x) | 149.6 ms |
| 64K | 17.9 ms | 1879 MB | 176.1 ms | 9.8x | 177.4 ms (9.9x) | 598.3 ms |

1. **The decode step is launch-bound and barely grows with context**
   (16 -> 18 ms from 4K to 64K), while the V volume grows linearly in S. So the
   transfer/step ratio *gets worse* with longer contexts; there is no
   "long context makes it worth it" crossover.
2. **The wall is PCIe, not NVDEC.** Decoding V through NVDEC every step is
   10-33x the step; and even if the decode were free, moving 1.88 GB of pixels
   into VRAM costs 176 ms. The NVDEC column is the decode alone, before any
   bitstream transfer.
3. **"V is the tough one, K the sensitive one" does not hold at 7 bits**
   (`--kv-split`, same calibrated profile): V-only -0.050 ppl, K-only -0.078,
   both +0.003; at 6 bits V-only +0.164, K-only -0.022. The K/V asymmetry is a
   4-bit phenomenon (see the llama.cpp section); at usable depths both streams
   are equally cheap.

So: **at <=4K, V offload costs ~10%** (acceptable, but it only saves 235 MB of
VRAM); **from 8K it exceeds the step (2.9x at 16K, 9.9x at 64K)**. The right move is to keep V
in VRAM at 7 bits (45%, zero PCIe, +0.03 ppl), not to move it to host memory or
a video stream.

The only defensible use: land V on **disk** for cold start / cross-machine
transfer (`save_archive`) and decode once in the background — never on the
per-step path.

## Repository layout

```
flashcodec/
├── flashcodec/
│   ├── __init__.py        # apply_flash_codec / CodecKVCache / JPEGCodec / PNGCodec
│   ├── codecs/jpeg.py     # DCT-II basis, project/reconstruct (pure GEMM, no loops)
│   ├── codecs/png.py      # anchor + first-order delta, cumsum, Top-M filter
│   ├── cache.py           # CodecKVCache: raw tail + capacity-doubled archives
│   ├── calibrate.py       # per-layer/stream channel profile (one forward pass)
│   ├── storage.py         # save_archive / load_archive (zlib/zstd, disk & transport)
│   ├── export.py          # export_model / load_pretrained (7-bit calibrated bundle)
│   ├── attention.py       # materialization-free decode attention (coefficient domain)
│   └── patch.py           # one-line hot patch: cache-level injection (HF Cache duck-typing)
├── tests/                 # jpeg / png / causal / attention / calibrated-bits / storage / export
├── benchmarks/            # storage, PSNR/cosine, ppl+calibration, offload/NVDEC, MLA
├── docs/                  # measured comparisons (vs llama.cpp KV quantization)
├── demo/                  # Gradio side-by-side chat
├── images/                # README screenshots
├── .github/workflows/     # CI: CPU tests + transformers integration
├── CHANGELOG.md
├── pyproject.toml
└── README.md
```

## Tests

```bash
python tests/test_jpeg.py           # orthonormality, k=r lossless, k=2 cosine>0.99, K monotone
python tests/test_png.py            # lossless maxdiff<1e-6, variance ratio printed, differentiable
python tests/test_causal.py         # zero future leakage, incremental vs full ≤1e-4,
                                   # memoized read == full decode, exact + repeatable crop,
                                   # beam-search batch gather
python tests/test_hf_integration.py # native transformers 5.x generate: explicit cache / hot patch /
                                    # PNG lossless exact tokens / 70-token multi-block stability /
                                    # beam search + crop exactness against the unpatched model /
                                    # per-layer schedule
python tests/test_attention.py      # coefficient-domain decode attention matches the materialized
                                    # path (jpeg / int8 / png, GQA); never builds the KV view
python tests/test_calibrated_bits.py # bit-packing round trip (2..8 bits), calibrated profile lowers
                                    # error, packed archives stay causal, Triton == torch
python tests/test_storage.py        # save/load bitwise round trip (CodecKVCache + FlashCodecCache),
                                    # compressed smaller than raw, multi-layer needs a config
python tests/test_export.py         # manifest round trip, profile reload, patched generation,
                                    # archive ratio
```

## Stability (1.x)

The public API is frozen for 1.x: `apply_flash_codec` / `remove_flash_codec`,
`apply_codec_attention` / `remove_codec_attention`, `FlashCodecCache`,
`FlashCodecLayer`, `CodecKVCache`, `JPEGCodec`, `PNGCodec`,
`codec_decode_attention`, `make_codec`, plus the `quant_bits` / `per_layer`
options. New features ship as additive arguments; existing signatures keep
their defaults. The caveats below still apply — this is a research library with
a stable interface, not a claim of production readiness.

## Boundaries (honest list)

- Research prototype: `read()` materializes the full reconstructed view for
  attention; memory gains are at the **cache storage layer** (table above);
  end-to-end benefits require combining with materialization-free kernels
  (flash-attention style). That view is memoized and reused across decode
  steps, so steady-state memory = archive + one raw-KV-sized view.
- Full-sequence `forward(seq)` prefill compresses complete blocks directly, so
  mid-block attention sees the block-level reconstruction (within-block
  approximation); the incremental `decode_step` path has no such issue.
- Compressed tensors are dense layouts; PNG Top-M sparsity is not yet wired to
  an entropy coder. (Packing sparse png alone was measured to be a wash: the
  fp32 delta values dominate, and on bf16 input the packed form can be *larger*
  than raw — delta quantization has to come first.)
- Coefficient truncation (`k < r`) is a per-layer decision: on a dense model
  every layer is compressed and the error compounds (measured: Qwen3-0.6B
  perplexity 16.8 → 6486 at k=2). Use `k = r` + `quant_bits=8` there (~52%
  storage, +2% ppl) and reserve aggressive ratios for hybrid models where only
  the full-attention layers carry a codec.
- **Not a better KV quantizer than llama.cpp's — as an archive format.** At
  equal bytes (truncation, and lossless basis + `quant_bits=8`: +0.05% vs
  +0.38% ppl); 4-bit V is nearly free while 4-bit K is the only cliff. The
  *transform* is another matter: give the coefficients ggml's per-32 absmax and
  it wins 5×/7× at 8/4 bits with the same reconstruction error — measured in HF,
  not yet confirmed in llama.cpp. See
  [`docs/llama-cpp-kv-quantization.md`](docs/llama-cpp-kv-quantization.md).
- Same lineage as the v4.2 prototypes inside
  [Mini-Architecture](https://github.com/aifeifei798/Mini-Architecture)
  (there: per-position view semantics for cross-layer KV reuse; here:
  standard cache storage semantics).

## License

[MIT](LICENSE)
