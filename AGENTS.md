# AGENTS.md — flashcodec

Standalone open-source project: compresses Transformer time-series KV cache tensors
as "high-dimensional multimedia image signals" (jpeg = frequency-domain DCT lossy /
png = time-domain delta lossless). Pure torch, no pytest dependency, no CI.
Read this whole file before making changes.

## Running conventions

- **You must run from inside this directory (`flashcodec/`)**: the package and the
  project directory share a name, so `import flashcodec` from the parent directory
  only hits the namespace package (no `__version__`/exports) — an inherent trap of
  this same-name layout. After `pip install -e .` it imports from anywhere.
- Tests are plain runnable scripts (`assert` + `__main__` summary, function names
  are pytest-compatible): `python tests/test_jpeg.py`, `python tests/test_png.py`,
  `python tests/test_causal.py`, `python tests/test_hf_integration.py`,
  `python tests/test_calibrated_bits.py`, `python tests/test_storage.py`.
  Run all six after touching codecs/cache.
- Benchmarks: `python benchmarks/bench_reconstruct.py` (PSNR/cosine),
  `python benchmarks/bench_vram.py [--cuda]` (storage occupancy/timings).
- This directory is gitignored inside the host repo — it is an independent git
  repository with independent releases.

## transformers integration (only targets local 5.20; 4.x explicitly unsupported)

- `FlashCodecLayer(DynamicLayer)` + `FlashCodecCache(Cache,
  layer_class_to_replicate=partial(...))`: 5.x's Cache is a layer container, and
  the base class lazily creates layers and hands them down to the layer; **5.x
  attention consumes the return value of `update` directly** (it does not read
  `key_cache`), so layer.update must return the reconstructed full causally-visible
  KV.
- `apply_flash_codec`'s main hook is `model.generate` (5.x generate builds its own
  DynamicCache internally, so injection only takes effect at the entry point); the
  forward wrapper is secondary.
- Generation semantics: **cache length = prompt + new_tokens - 1** — the last
  generated token's KV is never computed (nobody attends to it), so don't write the
  assertion as the total length.
- Integration-test traps: a randomly initialized tiny model will emit eos and stop
  early (disable it with `generation_config.eos_token_id = None`); scaling lm_head
  by 20 creates a clear argmax margin, isolating tie jitter and focusing on the
  cache mechanism.
- `crop` (speculative decoding path) is exact: archived blocks are kept
  verbatim, only the boundary partial block returns to the raw tail (no
  re-compression, so repeated crops are idempotent); `reorder_cache` /
  `batch_repeat_interleave` / `batch_select_indices` gather the archive + tail too
  (beam search is bitwise-identical on the lossless tier); `offload/prefetch` are
  still no-ops.
- `_CodecStream.read()` is **memoized**: decoded archive blocks are kept in a
  persistent append-only `_rec` buffer (one decode per new block instead of
  re-decoding the whole history per decode step — measured 38.7ms → 5.2ms at
  S=64K in bench_vram). Two consequences not to "optimize away": (1) `read()`
  returns a **view** into `_rec` when the raw tail is empty, so structural ops
  must not write the live region in place (`select_batch` allocates a new buffer;
  `reset()` is the explicit invalidation point); (2) the reconstruction buffer is
  persistent, so steady-state memory includes one raw-KV-sized view (before, it was
  a per-read temporary — peak is unchanged, steady state is higher).
- **jpeg int8 (`quant_bits=8`)**: coefficients are int8 plus a per-(block,
  frequency) fp32 scale (`_scales (B,H,cap,k,1)`); decode = dequantize → DCT.
  Real byte saving (fp32 archive 25% → 6.6%, bf16 archive 25% → ~13%) at ~0.3 dB
  PSNR and +2% ppl on a dense model. png rejects `quant_bits` on purpose (the
  lossless tier must keep fp32 deltas).

## dtype contract (distilled from Qwen3.5-0.8B practice; read before changing codecs)

- **PNG deltas are always fp32**: computing the deltas of bf16 inputs in bf16 loses
  ~0.4% precision, and cumsum reconstruction is no longer bitwise-exact (bf16
  deltas are exactly representable in fp32, so anchor+ΣΔ restores the original
  values bitwise) — this is the basis for "lossless → generated tokens are
  bitwise-identical to the baseline". Under bf16 deployment the archive is about
  1.9× dense (bandwidth traded for exactness).
- **JPEG coeffs are stored in the input dtype** (computed in fp32, then cast):
  lossy modes need no higher precision, and the storage ratio isn't inflated
  (k=2 measured ~27% raw).
- **`_CodecStream.read()` uniformly restores the stream's input dtype**: SDPA
  requires q/k/v to share a dtype, so feeding fp32 views to a bf16 model crashes
  outright.
- Hybrid architecture (Qwen3.5: 18 linear_attention + 6 full_attention): you must
  use `FlashCodecCache.from_config(model.config)` to attach a codec only to
  full_attention layers per layer_types, leaving the other layers as native dynamic
  layers (recursive state and other interfaces are unaffected);
  `apply_flash_codec` already routes through from_config automatically.
- Practical script: `benchmarks/run_qwen_demo.py` (PNG lossless bitwise-identity
  assertion + JPEG storage-ratio/fluency report + hybrid-layer audit).

## Tensor contract (read before changing any signature)

- KV layout `(B, H, S, D)`; **the compression/filtering axis is always dim=-2 (the
  seq axis)**.
- Codecs may only use batched matmul/elementwise operators; **explicit Python for
  loops are forbidden**; the trailing partial block is zero-padded
  (`_pad_to_blocks`) and trimmed back via `out_len`.
- Codecs compute in fp32 uniformly (`dct_compress` and friends call `.float()`
  internally); DCT with keep=r and PNG with keep_ratio=1 are the "lossless"
  baselines (fp round-trip on the ~1e-6/1e-7 level; see the tests for assertion
  tolerances, and don't tighten them to bitwise).

## Causal-safety invariants (breaking these is a bug)

1. **Archive complete blocks only**; an incomplete block always stays in
   `raw_buffer` (original precision) — a future perturbation at any point during
   incremental decoding must never rewrite already-archived tensors (`test_causal`
   verifies this bitwise).
2. The two fill paths — full `forward(seq)` and stepwise `decode_step` — must
   produce **bitwise-identical archives** (compression is per block and grouping
   invariant) and views that agree exactly on png (cumsum is per-block) and within
   ≤1e-6 on jpeg (batched-GEMM reduction order depends on the batch size once
   `read()` is memoized: one block vs N blocks; on some BLAS builds it is bitwise,
   on others not — don't assert `torch.equal` for jpeg).
3. In the `mode="png", keep_ratio=1.0` lossless mode, stepwise decoding attention
   must align with the full forward pass within ≤1e-4 (the tests' correctness
   anchor).

## Evaluation findings (measured; don't re-litigate without data)

- **Real post-RoPE KV is not uniformly smooth.** `Var(Δ)/Var(X)` along seq:
  K ≈ 0.16–0.72, V ≈ 0.09–1.87 across Qwen3-0.6B / Qwen3.5-0.8B layers (values
  are often *anti*-correlated). The 30× variance-compression number in
  `test_png` is a property of the synthetic random walk — keep
  `bench_reconstruct` labelled synthetic and don't cite it as a real-KV claim.
- **Truncation compounds per layer.** Dense Qwen3-0.6B (28/28 layers): ppl
  16.79 → 470 (k=4) → 6486 (k=2). Hybrid Qwen3.5-0.8B (6/24 layers): 22.48 →
  22.30 (k=4) → 24.06 (k=2). Safe default for dense models is **k = r +
  `quant_bits=8`**: ~52% of bf16 storage, ppl 16.79 → 17.13 (+2%).
- **Sparse-png packing alone is a wash** (fp32 delta values dominate; bf16 input
  makes the packed form larger). Delta quantization would have to come first —
  don't ship bit-packing without it.
- **End-to-end overhead is small at short context** (Qwen3.5-0.8B, 48 new
  tokens: baseline 47.0 tok/s vs 43.5–49.5 tok/s with codec tiers, peak +30MB),
  but peak memory ≈ raw KV + archive because `read()` materializes a view;
  memory wins need materialization-free kernels (flash-attention style).

## Materialization-free decode attention (`flashcodec/attention.py`)

- Both codecs are linear along seq: the r tokens of a block are `A @ S` (jpeg:
  `A = M^T` (r,k), `S` = coeffs or dequantized int8; png: `A = tril(ones)`,
  `S = [anchor; deltas]`). Therefore `q·K[p] = (q·S^T) A[p]` and
  `out = (prob·A) S` — decode attention without ever building the view.
  GQA is grouped over query heads; never `repeat_interleave` the stored rows.
- **Decode only** (`Sq == 1`); prefill would blow up the score tensor. png is
  numerically supported but stores r rows per block, so it saves nothing — the
  win is jpeg `k < r`. Remaining peak is the int8→fp32 dequantization of rows.
- Measured (B=1 H_kv=8 H_q=16 D=128, 64K, jpeg k=2 int8): materialized
  0.89ms/520MB → coefficient-torch 0.43ms/96MB → **coefficient-Triton
  0.17ms/0MB**. The Triton kernel is `flashcodec/triton_attention.py`
  (split-K + combine, flash-decoding style); `do_not_specialize` on
  nb/nr/chunk/ns is required, otherwise every new sequence length recompiles.
  `impl="auto"` picks Triton on CUDA and falls back to torch for png / CPU /
  D>256 / no blocks.
- **HF integration**: `apply_flash_codec(..., materialization_free=True)` wraps
  standard Llama/Qwen-style attention forwards (`patch_attention.py`); guards are
  batch=1 / single-token / eval / no sliding window / standard
  `position_embeddings`, otherwise it falls back silently. Prefill still
  materializes; the first decode step drops the view (`discard_view()` +
  `layer.keys = None`, otherwise the prefill view stays referenced and no memory
  is saved).
- End-to-end (Qwen3-0.6B, 4096-token prompt + 32 new): baseline 52.1 tok/s /
  peak 581MB; codec k=2 int8 materialized 44.2 tok/s / steady 1535MB; with
  materialization-free (Triton) 40.3 tok/s / peak 642MB / steady 122MB.
  Per decode step: raw 17.8ms, materialized 21.7ms, Triton 22.8ms — **12.6×
  less steady memory for ~9% slower decode**. A persistent kernel is the
  remaining optimization (the per-layer split/combine launch pair).
- `per_layer={idx: {keep_freqs/quant_bits/...}}` on `from_config` applies a
  calibrated schedule. Measured Qwen3-0.6B: adaptive (cheapest 14 layers at k=4)
  40.83 ppl at 75% storage vs uniform k=6 48.05 at 75% and uniform k=r int8
  19.36 at 50%. Per-layer damages are super-additive — truncation is a last
  resort, int8 is the default. Repro: `benchmarks/bench_adaptive.py`.

## Channel profile and sub-byte packing (1.1)

- **Scale model.** The archive stores one fp32 scale per (block, frequency) row,
  shared by all head_dim channels. Per-channel magnitudes differ by orders of
  magnitude, so a *static* calibrated profile `g[freq, d]` folded into the step
  (`step = scale * g`) cuts the reconstruction error at 8 bits (measured
  0.0224 -> 0.0059 on K, 0.0094 -> 0.0065 on V), and is what makes 7-bit
  usable. The profile is per (layer, stream) — K and V
  profiles differ enough that sharing one table is *worse* than no table on V.
  `FlashCodecLayer` therefore builds two codec instances; do not "share them for
  tidiness".
- **Calibration is offline and causal.** `calibrate_channel_scales` runs one
  forward on 1-2k tokens; the table is a training-time constant, never derived
  from the stream being decoded (that would break the causal invariant).
- **Measured ppl (Qwen3-0.6B dense, 16x512 windows built from the repo's
  markdown, disjoint 2048-token calibration, baseline 31.1504)**: b8 31.6745
  (+0.524) -> b8+profile 31.1210 (-0.029) -> b7+profile 31.1750 (+0.025) at
  45.3% -> b6+profile 31.3993 (+0.249) at 39.1%. Without the profile b7 is
  +2.34 and b6 +16.05, b4 collapses (ppl 11336). Hybrid Qwen3.5-0.8B (6/24
  layers, baseline 23.5318): b6+profile -0.010. Repro:
  `benchmarks/bench_calibrated.py`.
- **Bit packing** is an MSB-first contiguous stream, zero-padded to a byte
  boundary so any head_dim works (`archived_dim(D) = ceil(D*bits/8)`). The cache
  stores uint8 with that last dim; `head_dim` must be tracked on the stream
  (`_CodecStream.head_dim`) because `_coeffs.shape[-1]` is no longer D. The
  Triton kernel unpacks with a gather (`(b0<<8|b1) >> (16-p0-bits)`), which is
  ~2x the cost of the plain int8 load at 64K — acceptable, but do not expect it
  to be free.
- **`storage.py` is a storage/transport codec, not a hot-path one.** It expands
  packed coefficients back to int8 first (bit-shifted streams compress worse),
  concatenates a stream's parts and zlibs them. Loading re-packs; pass
  `device="cuda"` (CPU packing is ~40 Mval/s, GPU ~100x faster).

## Negative results (measured; do not re-propose without new data)

- **2D DCT across heads is useless.** At an equal coefficient budget (16 of 64)
  on real KV: keep all heads + 2 time freqs gives cos 0.970/0.815/0.768
  (layer 0 K / 14 V / 27 V); keep all time freqs + 2 head freqs gives
  0.826/0.524/0.524. The compressible energy is on the seq axis only. Any 2D
  scheme that truncates the channel/head axis loses.
- **Cosine is a mirage.** k=8 b=4 measures cos 0.994 and ppl 11336; k=8 b=6
  measures cos 0.9986 and ppl 43.3. Judge every quantizer change by ppl, never
  by cosine.
- **Entropy coding cannot go on the hot path.** Symbol entropy at 8 bits is
  6.84/8 (K 6.32, V 7.36) — ~15% of the payload — but decoding it every step
  costs more than it saves, and Huffman/ANS is serial. Disk only.
- **NVDEC / real-JPEG decode**: works (nvJPEG, 2.07 Gpix/s via torchvision) but
  a decode step needs 56 images (28 layers x K,V), 0.55 s encode + 0.11 s decode
  at 4K vs 3.6 ms for the Triton coefficient path (~183x), ~2200x at 64K. The
  codec is not a JPEG bitstream and making it one does not pay.

## Design decisions (don't "refactor these away" in passing)

- **The HF hot patch lives in the Cache layer, not the attention kernel**: KV
  writes/reads in standard HF models converge on the Cache interface
  (`update`/`key_cache`), so injecting a `CodecKVCache` duck type is enough — far
  more robust than per-model surgery (GQA/MQA/RoPE variants).
- **PNG dense storage honestly counts as 1.0**: anchor+deltas are the same size as
  the original block; what it saves is information entropy (variance compression
  ~30×), and that headroom is left to an entropy coder — `storage_ratio()` must not
  "round" this into (r-1)/r.
- A full `forward(seq)` prefill compresses complete blocks directly, so in-block
  position attention sees the block-level reconstruction (intra-block
  approximation); the incremental path has no such issue. This is documented
  semantics, not a bug.
- Archive data structure: **a capacity-doubling preallocated single tensor**
  (std::vector style, starting at 16 blocks, doubling when full) — jpeg uses
  `_coeffs (B,H,capacity,k,D)`; png uses `_deltas (B,H,capacity,r-1,D) fp32 +
  _anchors (B,H,capacity,D)`. Each full block only does an in-place slice write;
  `read()` decodes only the newly archived blocks (memoized `_rec`) and cats the
  raw tail.
  **Do not** revert to "append a new archive tensor per full block + loop over
  archives in read" — with that fragmentation, per-token decoding measured 170×
  slower reads (31ms → 0.18ms at S=4096, launch overhead dominating).
  `storage_bytes()` only counts the logical valid region, not the free capacity
  reserved by doubling. `reset()` zeroes the counters but **retains allocated
  capacity** (reused for crop/refill VRAM).

## Release

- `images/` holds the README screenshots and is maintained by the user — do not
  regenerate, replace, or commit new images unless explicitly asked.
- `pyproject.toml` (setuptools, MIT, torch>=2.0); the version number is written in
  `pyproject.toml`, `flashcodec/__init__.py::__version__` **and the top
  `CHANGELOG.md` entry**, so changing one requires syncing all three
  (`uv lock` afterwards: the lock records the project's own name/version).
- **Distribution name is `flashcodec-kv`, import name is `flashcodec`**; the GitHub
  repo is `aifeifei798/flashcodec-kv` (renamed from `flashcodec`; old URLs
  redirect). The bare `flashcodec` name on PyPI is an unrelated 0.0.0 placeholder
  package, never publish under it (rename only if the user confirms they own that
  name).
- `.github/workflows/ci.yml` runs the three codec tests plus the transformers
  integration test on CPU (Python 3.10 / 3.12, `pip install -e ".[hf]"`).
  `requires-python = ">=3.10"` is driven by transformers 5.20; don't lower it
  without also making the `hf` extra conditional, or `uv lock` fails.
- Before publishing to PyPI, run all four test groups + both benchmarks + 
  `benchmarks/run_qwen_demo.py` (the measured tables in the README must be updated
  accordingly — every number in the README was actually measured; no placeholders
  or stale data), then `uv build`.
- **1.x API is frozen**: `apply_flash_codec` / `remove_flash_codec`,
  `apply_codec_attention` / `remove_codec_attention`, `FlashCodecCache`,
  `FlashCodecLayer`, `CodecKVCache`, `JPEGCodec`, `PNGCodec`,
  `codec_decode_attention`, `make_codec`, `quant_bits`, `per_layer`,
  `channel_scale` / `channel_scale_v`, `calibrate_channel_scales`,
  `save_archive` / `load_archive`. Add options with defaults; breaking changes
  need a 2.0.
