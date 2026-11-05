# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The import name is always `flashcodec`; the PyPI distribution is
`flashcodec-kv` (the bare `flashcodec` name on PyPI is an unrelated 0.0.0
placeholder package).

## [1.1.0] - 2026-11-05

### Added

- **Calibrated channel profile** (`JPEGCodec(channel_scale=...)`,
  `calibrate_channel_scales(model, input_ids)`): one static
  `(keep_freqs, head_dim)` table per layer *and stream*, folded into the
  quantizer step (`step = scale[block, freq] * profile[freq, d]`). The table is
  a constant, so decode pays one broadcast multiply. Measured on Qwen3-0.6B
  (16 x 512-token windows built from the repository's own markdown, profile
  calibrated on a disjoint 2048-token slice, baseline ppl 31.1504):

  | tier | ppl | dppl | of bf16 |
  |---|---|---|---|
  | k=8 b=8 | 31.6745 | +0.524 | 51.6% |
  | k=8 b=8 + profile | 31.1210 | -0.029 | 51.6% |
  | k=8 b=7 | 33.4916 | +2.341 | 45.3% |
  | **k=8 b=7 + profile** | **31.1750** | **+0.025** | **45.3%** |
  | k=8 b=6 | 47.2046 | +16.05 | 39.1% |
  | k=8 b=6 + profile | 31.3993 | +0.249 | 39.1% |

  Hybrid Qwen3.5-0.8B (6/24 layers carry a codec, baseline 23.5318): b=7 +
  profile +0.015, b=6 + profile -0.010. Without the profile, b=6/b=4 stay
  catastrophic on the dense model (+16.05 / ppl 11336 for b=4).
  Reproduce: `benchmarks/bench_calibrated.py`.
- **Sub-byte coefficient packing** (`quant_bits` 2..8): contiguous MSB-first
  bit stream, byte-padded so any head_dim works; unpacked on the fly by both
  the torch path and the Triton split-K kernel (gather-based, no extra pass).
- **`save_archive` / `load_archive`** (`flashcodec/storage.py`): the storage /
  transport entropy coder. Packed coefficients are expanded back to int8 before
  compression (bit-shifted streams compress worse), one zlib blob per stream,
  zstd when importable. 28-layer Qwen3-0.6B, 4001 tokens, k=8 b=7 + profile:
  208.0 MB in memory -> 181.6 MB on disk (76.7% of the logical payload, 39.6%
  of the bf16 KV), bitwise round trip; save 5.0 s, load 13.2 s on CPU /
  1.46 s with `device="cuda"`.
- `tests/test_calibrated_bits.py` and `tests/test_storage.py` (both in CI).

### Changed

- `JPEGCodec.storage_ratio()` accounts for the bit depth.
- `FlashCodecLayer` / `FlashCodecCache` / `CodecKVCache` accept `channel_scale`
  and `channel_scale_v`; per-layer tables are passed through `per_layer=`.
- `FlashCodecLayer` builds separate key/value codec instances (they are
  stateless — this is what makes per-stream profiles possible without changing
  the archive layout).
- Decode cost at 64K (B=1, H_kv=8, H_q=16, D=128, k=8, per layer): b=8
  0.371 ms, b=8 + profile 0.414 ms, b=7 + profile 0.767 ms (the gather unpack
  is ~2x the plain int8 load; still far below the materialized path).

## [1.2.0] - 2026-11-05

### Added

- **Model bundles** (`flashcodec/export.py`): `export_model()` calibrates a
  checkpoint once and writes a `flashcodec-kv.json` sidecar holding the codec
  tier and the per-layer / per-stream channel profiles (base64 fp32);
  `load_pretrained()` and `apply_export()` restore them, so a 7-bit KV cache is
  active right after loading. Weights are untouched — this exports the
  *KV-cache* tier, not weight quantization. CLI:
  `flashcodec-export` / `python -m flashcodec.export`, plus `--inspect`.
  Measured on Qwen3-0.6B, 2048-token calibration: bundle 313 KB for 28 layers,
  45.3% of bf16 KV, and the generated tokens are identical to the uncompressed
  model on the same prompt.
- `tests/test_export.py` (CI): manifest round trip, profile reload, patched
  generation, storage ratio.

## [Unreleased]

### Added

- `benchmarks/bench_offload.py`: measures whether V can live on host RAM / a
  video stream. Warm pinned H2D is 10.7 GB/s (Gen4 x8); the decode step is
  launch-bound (16.1 -> 17.9 ms from 4K to 64K) while the V transfer grows
  linearly (11.0 / 44.0 / 176.1 ms), so the best-case pipelined step is
  1.1x / 2.9x / 9.9x, and NVDEC-per-step is 37 / 150 / 598 ms. `--kv-split`
  also shows there is no K/V tolerance asymmetry at 7 bits (V-only -0.050
  ppl, K-only -0.078, both +0.003). README gains a matching "why not offload"
  section in both languages; no code or API change.

### Documentation

- `docs/llama-cpp-kv-quantization.md`: measured same-bytes comparison against
  llama.cpp's KV quantization on Qwen3-0.6B (llama.cpp `184f65360`). Verdict:
  `-ctk q8_0` is ~free at 53% of fp16 bytes, while both usable flashcodec tiers
  lose at equal bytes (truncation needs 2.7× the bytes for the same damage;
  lossless basis + int8 is 8× worse at 50%). Also records the K/V asymmetry
  behind the recommended `K=q8_0 + V=q4_0` (40.6% of fp16, +0.28% ppl) and why
  4-bit K is the only cliff. README gains a matching section and Boundaries
  bullet; no code or API change.
- Same doc, §3: stacking the codec **on top of** llama.cpp's own KV
  quantization (64 × 512 wikitext windows). On `q8_0` the codec reclaims 0.8%
  (K) / 1.6% (K+V) of the bytes while the damage grows 1.6–2.3×; on `q4_0` a
  lossless-basis int8 archive is *larger* than the 4-bit payload it replaces
  (39.8% vs 28.1%). Truncation on top of either quantizer stays catastrophic:
  the two compression stages are not complementary.
- Same doc, §5: the coefficient-domain quantizer flips the result. Giving the
  DCT coefficients ggml's per-32 absmax (lossless basis, identical byte count)
  instead of this library's per-frequency-row scale reduces the perplexity
  damage 5× at 8 bits (fp16, 1024-token windows: +0.028 vs +0.138) and 7× at
  4 bits (ppl 41.6 vs 186.8) with the *reconstruction error unchanged* — the
  transform shapes the error into the slowly-varying subspace that softmax
  barely sees. HF-measured, not yet confirmed in llama.cpp; 4-bit K remains
  unusable on this model.
- Same doc, §6: the 4-bit cliff is a *small-model* property. On a local 27B
  hybrid (DrakIdol Qwen3.8-27B-Plus Q4_K, `qwen35`, 17 KV-bearing layers),
  `q4_0` KV costs +0.19% at 512 ctx, +0.39% at 8k and +0.20% at 32k (CIs
  ±0.06–0.10), where the 0.6B costs +236%. So the §5 win lives where the model
  is already broken: a mechanism result, not a deployment feature. Gemma4-12B is
  dense but has a 302 perplexity on wikitext (roleplay finetune), so its deltas
  are not usable without an in-domain corpus.
- Same doc, §7: Gemma4-12B (`gemma4`, 48 layers = 40 SWA@1024 + 8 global MQA)
  measured with `--kl-divergence` because its perplexity is not a valid metric
  (baselines 302 / 355 / 2.24 / 18.87 across corpora; quantization sometimes
  *lowers* ppl). Mean KL: 0.168 at `q8_0`, 0.230 (`q4_0` K), 0.225 (`q4_0` V),
  0.303 at `q4_0` K+V, top-1 agreement 89.1%. No cliff at 12B, but 4-bit is not
  free either. K and V contribute equally in KL even though ppl blames V 2×.
  Same protocol on the 27B hybrid: KL 0.0021 (`q8_0`) / 0.0035 (`q4_0`) — 86×
  less distortion, because only 17 of its 64 layers read a KV cache.
- Same doc, §8: HF-side KL pre-judgment across three models (mean KL(tier||f16),
  top-1 agreement, 8192 scored tokens, equal-byte pairs). The transform helps
  only where row quantization is already destructive — Qwen3-0.6B: KL 0.0168 ->
  0.0056 at 8 bits, 2.375 -> 1.028 at 4 bits — and is neutral at 8 bits (0.0002
  -> 0.0003) with **worse** 4-bit (0.0364 -> 0.0968) on Hy-MT2-1.8B, and neutral
  (0.0025 -> 0.0028) with worse 4-bit (0.279 -> 0.330) on Gemma4-E2B. Verdict:
  no operating point where the codec is worth shipping, so no llama.cpp
  implementation follows.
- Same doc, §8: adds a **plain dense Qwen3-8B** (36 layers, 8 KV heads, no SWA /
  hybrid / MQA) as the decisive row — the architecture the codec was designed
  against. `q8_0` KL 0.0009 / top-1 99.2% -> dct+q8 0.0005 / 99.1% (both
  negligible); `q4_0` 0.0802 -> dct+q4 0.0852 (6% worse); at the operating point
  that matters, `q8_0` K + `q4_0` V (40.6% bytes), KL is 0.0047 raw vs 0.0045
  with DCT on K (top-1 97.3% vs 97.2%), i.e. the transform is a no-op while
  spending the 4 bits on V instead of K is worth 10x. K is again ~18x more
  sensitive than V, so the K/V asymmetry is architectural, not a size effect.
- README.md / README.zh-CN.md: new top-level "research note" section stating the
  negative result up front (what was tested, the equal-byte table, the operating
  point where the transform is a no-op, and the conclusion that it is not
  implemented in llama.cpp), linking to the full doc.
- Same doc, appendix: why the original "JPEG for weights" idea needs a rotation
  rather than coefficient truncation (weights have no autocorrelated axis and a
  flat trained spectrum; the measured KV analogue is that truncation costs 2.7x
  the bytes of block quantization for the same damage), plus the one-afternoon
  comparison that would settle it. The appendix also states the two conditions
  the JPEG analogy needs — energy concentration (`Var(Δ)/Var(X) = 2(1−ρ)`, i.e.
  ρ ≈ 0.64–0.92 for text K vs ρ ≈ −0.07–0.82 for V) and a non-adaptive baseline
  quantizer (ggml's per-32 absmax is already an adaptive table) — and why the
  damage sits along `head_dim` rather than in the frequency domain, making a
  rotation there (QuaRot/SpinQuant/`hdmd`) the correct form of the same idea.

## [1.0.0] — 2026-10-23

First stable release. The public surface is frozen for the 1.x line (breaking
changes need a 2.0); the research caveats in the README's *Boundaries* still
apply, and every number in the README is measured.

### What 1.0 covers

- **Codecs**: `jpeg` (blockwise DCT-II, keep k of r, optional int8 coefficients
  with per-(block, frequency) scales) and `png` (anchor + first-order deltas,
  bitwise lossless at `keep_ratio=1.0`).
- **Caches**: `CodecKVCache` (standalone, no transformers) and
  `FlashCodecCache` / `FlashCodecLayer` (native `transformers` 5.x Cache
  backend, hybrid architectures via `from_config`, per-layer budgets).
- **Integration**: `apply_flash_codec` / `remove_flash_codec` hot patch; beam
  search (`reorder_cache`), exact `crop`, memoized `read()`.
- **Decode attention**: coefficient-domain `codec_decode_attention` (no KV view)
  plus the Triton split-K kernel, wired into `generate` with
  `materialization_free=True` for standard Llama/Qwen/Qwen3.5 attention.
- **Benchmarks**: `bench_reconstruct`, `bench_vram` (CUDA-event timed),
  `bench_attention`, `bench_e2e`, `bench_ppl`, `bench_adaptive`,
  `run_qwen_demo`.
- **Demo**: five tiers, materialization-free toggle, and a full per-tier
  comparison (storage / speed / memory / continuation NLL / token match).

### Stability

The 1.x public API: `apply_flash_codec`, `remove_flash_codec`,
`apply_codec_attention`, `remove_codec_attention`, `FlashCodecCache`,
`FlashCodecLayer`, `CodecKVCache`, `JPEGCodec`, `PNGCodec`,
`codec_decode_attention`, `make_codec`, `quant_bits`, `per_layer`. New options
are additive; existing signatures keep their defaults.

Published to PyPI as [`flashcodec-kv` 1.0.0](https://pypi.org/project/flashcodec-kv/)
(import name stays `flashcodec`).

## [0.6.1] — 2026-10-23

### Fixed

- **`materialization_free=True` on Qwen3.5**: its attention uses a Q-gate
  (`q_proj` emits query + gate, and the attention output is scaled by
  `sigmoid(gate)` before `o_proj`). The wrapper now handles that layout and
  guards on `head_dim` / `o_proj` width, falling back to the original forward for
  anything that does not match (no more shape errors on hybrid models).

### Changed

- Demo comparison is much more detailed: five tiers (png lossless, jpeg k=8
  int8, k=4, k=2, k=2 int8), a materialization-free toggle, and a "Full
  comparison" tab that reports per tier — KV footprint vs raw, archive/view
  split, decode tok/s, peak/steady memory, continuation NLL (perplexity-style)
  and token match, with storage/speed/NLL bars.

## [0.6.0] — 2026-10-23

### Added

- **Triton split-K kernel for coefficient-domain decode attention**
  (`flashcodec/triton_attention.py`, flash-decoding style): block chunks write
  partial (max, sum, acc), a combine kernel merges them. Dequantization, scoring,
  online softmax and the output accumulation happen in registers, so no
  intermediate tensors are allocated. One layer at 64K (`jpeg k=2 int8`,
  B=1, H_kv=8, H_q=16, D=128): materialized 0.89ms/520MB → torch coefficient
  0.43ms/96MB → **Triton 0.17ms/0MB**.
- `codec_decode_attention(..., impl="auto"|"torch"|"triton")`; auto picks Triton
  on CUDA and falls back for png / CPU / D>256 / empty archives.
  `do_not_specialize` on the dynamic scalars avoids per-step recompilation.
- End-to-end (Qwen3-0.6B, 4096-token prompt + 32 new): steady KV memory
  1535MB → 122MB (12.6×) at 40.3 tok/s vs 44.2 materialized (was 30.7 before the
  kernel); per decode step 22.8ms vs 21.7ms materialized / 17.8ms raw.
- `tests/test_attention.py` covers Triton-vs-torch agreement and dtype.

## [0.5.0] — 2026-10-23

### Added

- **HF integration for materialization-free decode**:
  `apply_flash_codec(..., materialization_free=True)` wraps standard
  Llama/Qwen-style attention forwards so decode steps attend on the stored rows
  (projection + RoPE unchanged, guarded to batch-1 / single-token / eval /
  non-sliding layers, silent fallback otherwise). The prefill view is dropped on
  the first decode step. `remove_flash_codec` restores everything;
  `codec_attention_stats(model)` reports the wrapped module count.
- `benchmarks/bench_e2e.py`: long-context end-to-end benchmark (speed, peak and
  steady memory). Qwen3-0.6B, 4096-token prompt + 32 new tokens:
  baseline 53.2 tok/s / peak 581MB; codec k=2 int8 materialized 44.9 tok/s /
  steady 1535MB; materialization-free 30.7 tok/s / peak 642MB / steady 122MB
  (12.6× less steady memory, ~1.7× slower decode).
- `tests/test_hf_integration.py` covers the patched generate path (lossless tier
  still token-identical, no reconstruction view retained, clean removal).

## [0.4.0] — 2026-10-23

### Added

- **`flashcodec/attention.py` — materialization-free decode attention.** Both
  codecs are linear along the seq axis (`K_block = A @ S`), so a decode query
  can attend on the stored rows: `q·K[p] = (q·S^T) @ A[p]` and
  `out = (prob @ A) @ S`. Measured at 64K with jpeg k=2 int8: 0.86ms / 520MB
  (materialized) → 0.46ms / 128MB (coefficient domain), maxdiff ≤ 1e-6.
  Decode only; png is numerically supported but has no memory win.
- **`per_layer` budgets** on `FlashCodecCache.from_config`, plus
  `benchmarks/bench_adaptive.py` (per-layer sensitivity calibration).
  Measured on Qwen3-0.6B: adaptive 40.83 ppl at 75% storage vs uniform k=6
  48.05 at 75% and k=r int8 19.36 at 50% — calibration helps over uniform
  truncation, but int8 remains the default because the damages are
  super-additive.
- `tests/test_attention.py` and a `benchmarks/bench_attention.py` memory/latency
  benchmark.

## [0.3.0] — 2026-10-23

### Added

- **`quant_bits=8` for the jpeg codec**: coefficients are stored as int8 with a
  per-(block, frequency) fp32 scale. A real byte-level saving on top of the
  `k/r` geometric ratio: 6.6% of fp32 raw KV at k=2, ~52% of bf16 raw KV at
  k=8 (full lossless DCT basis) with +2% perplexity on a dense model.
- **`benchmarks/bench_ppl.py`**: perplexity evaluation on a local corpus, run
  against a dense model (Qwen3-0.6B) and a hybrid one (Qwen3.5-0.8B).
- **Honest measurement fixes**: `bench_vram --cuda` now uses CUDA events plus a
  warmup (the old numbers were async and meaningless), and `run_qwen_demo`
  reports tokens/s and peak memory.
- README (EN/ZH): real-KV statistics and perplexity tables.

### Changed

- README claims about autocorrelation and "30× variance compression" are now
  qualified with measured real-model statistics; the dense-model caveat
  (coefficient truncation compounds per layer) is documented, with
  `k = r + quant_bits=8` as the safe tier for fully compressible models.

## [0.2.0] — 2026-10-23

### Added

- **Beam search support**: `FlashCodecLayer` / `CodecKVCache` implement
  `reorder_cache`, `batch_repeat_interleave` and `batch_select_indices` over
  the archives *and* the raw tail. On the lossless tier (`png keep_ratio=1.0`)
  `generate(num_beams>1)` is bitwise-identical to the unpatched model.
- **Exact `crop`** for speculative decoding: archived blocks are kept verbatim
  and only the boundary partial block returns to the raw tail, so repeated
  crops no longer re-compress the history (the old rebuild-from-view path added
  loss on every crop).
- **Memoized `read()`**: decoded archive blocks land in an append-only
  reconstruction buffer, so a decode step decodes one new block instead of the
  whole history (`bench_vram` S=64K, jpeg k=2: 38.7ms → 5.2ms).

### Changed

- Distribution renamed to `flashcodec-kv` (import name unchanged) because the
  `flashcodec` name on PyPI is taken by an unrelated placeholder package. The
  GitHub repository moved to
  [aifeifei798/flashcodec-kv](https://github.com/aifeifei798/flashcodec-kv)
  (the old URL redirects).
- Four new regression tests: memoized read vs one-shot decode, exact and
  repeatable crop, beam-search batch gather, and beam search / crop parity
  against the unpatched model.

### Fixed

- README numbers that had gone stale: the Qwen3.5-0.8B table now matches a
  fresh `benchmarks/run_qwen_demo.py` run (jpeg k=2 → 32.3% storage / 38.1%
  token match, jpeg k=4 → 54.8% / 49.2%, png lossless → 1.8× with bitwise
  identical tokens), and the CPU latency table reflects the memoized read.

## [0.1.0] — 2026-10-23

### Added

- Initial release: JPEG-style blockwise DCT-II codec (`mode="jpeg"`, keep the
  lowest K coefficients) and PNG-style anchor + first-order-delta codec
  (`mode="png"`, lossless at `keep_ratio=1.0`).
- `CodecKVCache` (standalone, duck-typed as a cache), `FlashCodecCache` /
  `FlashCodecLayer` (first-class `transformers` 5.x Cache backend, hybrid
  architectures handled through `from_config`), and `apply_flash_codec` /
  `remove_flash_codec` one-line hot patch.
- Causal-safety guarantees (complete blocks only; archived tensors are never
  rewritten), capacity-doubling preallocated archive storage, benchmarks
  (`bench_reconstruct`, `bench_vram`, `run_qwen_demo`) and the Gradio
  side-by-side demo.
