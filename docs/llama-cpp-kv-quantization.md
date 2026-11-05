# Same-bytes comparison with llama.cpp KV quantization

Measured 2026-10-23 on **Qwen3-0.6B** (28 layers, 8 KV heads, head_dim 128, i.e.
112 KiB of KV per token in fp16), RTX 5090 D, llama.cpp `184f65360`
(CUDA + flash-attention enabled), wikitext-2-raw.

## TL;DR

llama.cpp's own `-ctk q8_0` reaches **~53% of fp16 KV bytes at +0.05% perplexity**.
Two usable flashcodec **archive** tiers lose against it at equal bytes:

| flashcodec tier | bytes | damage at equal bytes vs `q8_0` |
|---|---|---|
| truncation (`k < r`) | 25–75% | **2.7× more bytes for the same damage**; `k=2` is unusable |
| lossless basis + `quant_bits=8` | 50% | **8× more damage** (+0.38% vs +0.05%) |

Stacking the codec on top of llama.cpp's quantizer does not help either: on
`q8_0` it reclaims 0.8% of the bytes for 63% more damage (§3).

**But the loss is the coefficient quantizer, not the transform.** The archive
above stores one fp32 scale per frequency row (128 values). Give the coefficients
the same per-32 absmax that ggml uses — identical bytes — and the ordering flips:
**5× less perplexity damage at 8 bits and 7× less at 4 bits (§5)**, with the
*reconstruction error unchanged*. The gain is error shaping (the error lands in
the slowly-varying subspace, which softmax barely sees), not a better K. That
result is HF-measured and still needs confirming inside llama.cpp; it does not
make 4-bit K usable on this model (ppl 41.6 vs 18.6).

**Scale check.** On a 27B hybrid (`qwen35`, 17 KV-bearing layers) the same
`q4_0` KV config costs +0.39% at 8k context and +0.20% at 32k — the small-model
cliff (+236% on 0.6B) does not exist at production scale (§6), and the 4-bit KL
there is 0.0035, so there is nothing to fix. On the 0.6B it is already broken.
Both ends leave the transform without a job; only the dense mid-size case was
still open, and §8 closes it.

**Dense mid-size models are where 4-bit KV is costly but not fatal** — and where
perplexity stops working as a metric. Gemma4-12B (48 layers, 40 SWA + 8 global
MQA) measured with `--kl-divergence`: KL 0.17 at 8 bits, 0.30 at 4 bits, top-1
agreement 89%. Its ppl ranges from 2.2 to 355 depending on the corpus and
sometimes *improves* under quantization, so KL is the only usable metric there.

**But the HF-side KL pre-judgment kills the follow-up** (§8): at equal bytes the
transform helps only where row quantization is already destructive
(Qwen3-0.6B: 3× lower KL at 8 bits), is neutral at 8 bits / slightly worse at
4 bits on Hy-MT2-1.8B and Gemma4-E2B, and on a **plain dense Qwen3-8B** — the
most favourable case — it changes nothing at the operating point that matters
(`q8_0` K + `q4_0` V, 40.6% bytes: KL 0.0047 raw vs 0.0045 with DCT on K) while
being 6% worse at 4 bits on both sides. There is no operating point where it is
worth shipping, so **no llama.cpp integration follows from this**.

Finally, K and V are not equally sensitive, which changes how a KV budget should
be spent.

## 1. llama.cpp KV types (the target to beat)

`llama-perplexity -c 512 --chunks 40 -fa on -ngl 99`, perplexity:

| K | V | ppl | Δ vs f16 | B/token | of fp16 |
|---|---|---|---|---|---|
| f16 | f16 | **19.6037** | — | 114,688 | 100% |
| q8_0 | q8_0 | 19.6138 | +0.05% | 60,928 | 53.1% |
| q8_0 | f16 | 19.6141 | +0.05% | 87,808 | 76.6% |
| f16 | q8_0 | 19.6080 | +0.02% | 87,808 | 76.6% |
| q8_0 | q4_0 | 19.6579 | +0.28% | 46,592 | 40.6% |
| f16 | q4_0 | 19.6606 | +0.29% | 73,472 | 64.0% |
| q4_0 | f16 | 66.0552 | +237% | 73,472 | 64.0% |
| q4_0 | q8_0 | 66.5416 | +239% | 46,592 | 40.6% |
| q4_0 | q4_0 | 65.8321 | +236% | 32,256 | 28.1% |

Throughput (`llama-bench -p 4096 -n 32 -fa 1`, 3 reps):

| KV | pp4096 t/s | tg32 t/s |
|---|---|---|
| f16 | 68,121 | 713.7 |
| q8_0 | 61,857 | 610.5 |
| q4_0 | 62,022 | 610.7 |

Two asymmetries worth remembering:

- **4-bit K is the only cliff**: K alone at `q4_0` gives 66.06 (+237%), while V
  alone at `q4_0` costs +0.29%. If a KV budget has to be squeezed,
  `K=q8_0 + V=q4_0` is the cheap end point (40.6% of fp16, +0.28% ppl).
- 8-bit is free on both sides, so the *only* open design space in llama.cpp is
  K below 4 bits — precisely where the codec is weakest.

## 2. Same-bytes A/B: seq-axis DCT vs row-wise quantization

Same model, `transformers` 5.x, bf16, 64 × 512-token wikitext windows.
`q8_0 emul` = symmetric absmax quantization per 32 contiguous head dims
(llama.cpp semantics, not bit-exact). `dct` = this project's blockwise DCT-II
along the sequence axis, K-only or K+V.

| tier | ppl | Δ | of fp16 |
|---|---|---|---|
| baseline | 27.9690 | — | 100% |
| q8_0 K+V | 28.2485 | +1.00% | 53.1% |
| q8_0 K only | 27.9823 | **+0.05%** | 76.6% |
| dct b32 k32 int8 K only | 28.0750 | +0.38% | 75.8% |
| dct b16 k16 int8 K only | 28.1194 | +0.54% | 75.8% |
| dct b8 k8 int8 K only | 28.3300 | +1.29% | 75.8% |
| dct b8 k8 int8 K+V | 28.3241 | +1.27% | 51.6% |
| dct b8 k=4 K only | 172.26 | +516% | 75.0% |
| dct b8 k=4 K+V | 649.6 | +2222% | 50.0% |
| dct b8 k=2 K only | 3142 | +11100% | 62.5% |
| dct b8 k=2 K+V | 21257 | +75900% | 28.1% |

Absolute perplexities in the two tables are not comparable (HF runs disjoint
windows in bf16, llama.cpp runs contiguous chunks in fp16); only the relative
damage is.

## 3. Stacking the codec on top of llama.cpp's quantization

If llama.cpp already stores quantized KV, the natural follow-up is whether the
codec can compress *those* values further. Same harness (64 × 512 wikitext
windows): the row quantizer runs first, the codec sees the dequantized rows, and
the byte count is the codec archive alone — the quantized values are replaced,
not stored twice.

| stack | ppl | Δ vs f16 | of fp16 |
|---|---|---|---|
| f16 | 27.9690 | — | 100% |
| `q8_0` K+V | 28.2485 | +1.00% | 53.12% |
| `q8_0` + dct b32/k32 int8 on K | 28.4259 | +1.63% | 52.34% |
| `q8_0` + dct b8/k8 int8 on K | 28.5795 | +2.18% | 52.34% |
| `q8_0` + dct b8/k8 int8 on K+V | 28.6244 | +2.34% | 51.56% |
| `q8_0` + dct b8/k4 on K | 160.59 | +474% | 51.56% |
| `q8_0` + dct b8/k2 on K | 2763.8 | +9781% | 39.06% |
| `q4_0` K+V | 201.95 | +622% | 28.12% |
| `q4_0` + dct b8/k8 int8 on K | 211.70 | +657% | 39.84% |
| `q4_0` + dct b8/k4 on K | 1195.1 | +4173% | 39.06% |
| `q4_0` + dct b8/k2 int8 on K | 5831.6 | +20751% | 20.51% |
| `q4_0` + dct b8/k2 int8 on K+V | 42085 | +150371% | 12.89% |

Reading:

- **On top of `q8_0` the codec can only reclaim the scale overhead** — one
  fp32 scale per 128 coefficient values instead of one fp16 scale per 32 raw
  values. That is 0.78% of the total bytes on K (52.34% vs 53.12%) and 1.56% on
  K+V, while the damage grows 1.6–2.3×. Strictly dominated.
- **On top of `q4_0` it is worse than useless.** An int8 coefficient archive is
  ~1 B/value, i.e. *larger* than the 0.5625 B/value it replaces, so the
  lossless-basis stack actually costs **more** bytes (39.84% vs 28.12%) for a
  worse model. Only aggressive truncation gets back under 4 bits, and those rows
  (20.5% at ppl 5831, 12.9% at ppl 42085) are broken models, not compressions.
- **The two stages are not complementary.** Both spend the same budget on the
  same information (K), and llama.cpp's per-32 absmax is already the cheaper way
  to spend it; putting an orthonormal transform in front only moves the error
  around and adds a second source of it.

The base quantizer here is an emulation in bf16 (symmetric absmax per 32 head
dims, applied to K and V), so its absolute damage is harsher than the real
`-ctk q8_0` (+1.0% vs +0.05% in §1: the emulation also quantizes V, which the
real configuration does not). The ordering, and every byte count, is what the
argument rests on.

## 4. Why the codec loses

1. **Truncation deletes variance, quantization only adds noise.** `dct k=4`
   (half the coefficients kept, 75% of fp16 K bytes) damages as much as
   `q4_0` (28% of fp16 K+V bytes). Attention logits are far more sensitive to a
   missing subspace than to a bounded perturbation. `k=2` is not a compression
   trade-off, it is a broken model. This matches the dense-model rows already in
   the README's perplexity table (16.8 → 470 → 6486), so it is not a
   llama.cpp-specific artifact.
2. **The gain from the basis is "decorrelate, then quantize" — and row
   quantization already has it.** Enlarging the block monotonically helps
   (b8 +1.29% → b16 +0.54% → b32 +0.38%), which is the signature of
   decorrelation, not of truncation. But ggml's per-32 absmax scale adapts
   locally every 32 values, while this codec stores one scale per
   (block, frequency) — i.e. one per 128 values. Larger blocks converge toward a
   global transform; they do not overtake the per-32 scale.
3. **K and V are not interchangeable.** V tolerates 4-bit almost for free; K
   does not even tolerate it at 8-bit-plus-margin. Compressing both with the
   same codec (which is what `FlashCodecCache` does) is therefore the wrong
   shape of design. This is the same conclusion that llama.cpp's DeepSeek-V4
   cache encodes structurally: `llama_kv_cache_dsv4` keeps a **compressed K-only**
   tier and leaves V alone.

## 5. Where the transform does win: the same quantizer on coefficients

> **v1.1 update.** The mismatch below — one fp32 scale per 128 coefficients vs
> ggml's per-32 absmax — is addressed in the archive itself by the calibrated
> channel profile (`JPEGCodec(channel_scale=...)`): a static
> `(keep_freqs, head_dim)` table folded into the step, per layer and stream.
> Measured effect at an identical byte count: up to ~4x lower reconstruction
> error on K (~1.4x on V) at 8 bits, and a 7-bit archive that costs +0.025 ppl (vs +2.34 uncalibrated) on
> Qwen3-0.6B. The analysis below is kept as-is because it isolates the *error
> shaping* effect of the transform, which the profile does not change.

§2–§4 tested flashcodec's *own* archive format, whose coefficient quantizer
stores one fp32 scale per (block, frequency) row, i.e. one per 128 values. That
is coarser than ggml's per-32 absmax. Feeding the same per-32 quantizer to the
coefficients instead (lossless basis, `k = r`, so the byte count is **identical**
to the same ggml type on raw rows) flips the result:

fp16 model, 1024-token windows scored on the last 512 (baseline ppl 18.59, close
to llama.cpp's 19.60):

| tier | ppl | Δ | of fp16 |
|---|---|---|---|
| f16 | 18.5913 | — | 100% |
| `q8_0` K+V (raw rows) | 18.7294 | +0.1382 | 53.12% |
| dct b8 + q8 K, q8 V | 18.6501 | +0.0588 | 53.12% |
| dct b16 + q8 K, q8 V | 18.6282 | +0.0369 | 53.12% |
| **dct b32 + q8 K, q8 V** | **18.6192** | **+0.0279** | **53.12%** |
| `q4_0` K+V (raw rows) | 186.83 | +168.24 | 28.12% |
| dct b16 + q4 K, q4 V | 64.33 | +45.74 | 28.12% |
| dct b32 + q4 K, q4 V | 47.83 | +29.24 | 28.12% |
| dct b32 + q4 K, q4 V (K only dct) | 41.61 | +23.01 | 28.12% |

At **identical bytes**: 5× less perplexity damage at 8 bits, and 7× less at
4 bits. Replicated on disjoint 512-token windows (`q8_0` +0.3595 vs
dct b32 +0.0536; `q4_0` 196.3 vs 46.8). Block 32 > 16 > 8 in both regimes.

### It is error shaping, not better reconstruction

Measured on real K rows (Qwen3-0.6B, layer 0/6/12/18/24), the **reconstruction
error is the same** for both quantizers:

| variant | rel-L2 | max abs err |
|---|---|---|
| raw q8 | 0.0111–0.0124 | 0.17–1.70 |
| dct b32 + q8 | 0.0110–0.0118 | 0.19–1.65 |

So the transform does not preserve K better; it moves the quantization error
into the low-frequency (slowly varying along positions) subspace. A logit offset
that is nearly constant across neighbouring positions is close to invisible to
softmax — only the error that changes the *relative* ordering of positions
costs perplexity. Hence the same L2 error and a much smaller ppl cost, and hence
also why larger blocks (finer frequency resolution) keep improving the result.

### What this does and does not change

- It **does not** rescue 4-bit: ppl 41.6 vs 18.6 is still a broken model, even
  though the damage is 7× smaller than `q4_0`.
- It **does** mean the seq-axis transform is a legitimate quantizer front-end
  (coefficient-domain steps matched to frequency), not just a truncation scheme.
  At 8 bits the absolute headroom is small because `q8_0` is already nearly
  free, so the value only shows up in regimes where the byte budget is tighter.
- It is measured in HF, not in llama.cpp, and this harness's absolute damage is
  larger than llama.cpp's real `q8_0` (+0.74% vs +0.05% at a comparable
  baseline). A llama.cpp implementation is needed before quoting it as a win
  there; a larger model is needed before claiming it makes 4-bit K usable.

## 6. Model size: the cliff is a small-model property

The same KV-type sweep on a production-scale local checkpoint — DrakIdol
Qwen3.8-27B-Plus, GGUF `Q4_K`, arch `qwen35`, 64 layers with
`full_attention_interval = 4` (17 KV-bearing layers), 4 KV heads × 256 dim,
i.e. 68 KiB of KV per token:

| context | f16 | `q8_0` K+V | `q4_0` K+V |
|---|---|---|---|
| 512 (20 chunks) | 6.8003 | 6.8038 (+0.05%) | 6.8130 (+0.19%) |
| 8192 (4 chunks) | 5.4791 | — | 5.5005 (+0.39%) |
| 32768 (4 chunks) | 6.1354 | 6.1384 (+0.05%) | 6.1475 (+0.20%) |

At 8k the split is K-only +0.24%, V-only +0.13%, and `q8_0` K + `q4_0` V is
+0.06% — all inside the ±0.10 CI. So the 4-bit regime that collapses the 0.6B
(+236%) is **essentially free at 27B**, and stays free out to 32k context.

Consequence: the cliff is a property of small models, not of KV quantization.
The DCT win of §5 sits exactly where the model is already broken, and the models
that actually have a byte-budget problem (large, long context) do not need it.
That leaves §5 as a mechanism result — the error shaping is real and reproducible
— rather than a deployment feature.

(Locally there is also Gemma4-12B, dense with 48 layers, but on wikitext its
perplexity is ~302 because it is a roleplay finetune; its deltas are dominated by
the ±23 confidence interval — e.g. `q4_0` K+V +13%, single-side +4%/0%. A
decisive dense mid-size test needs an in-domain corpus and a distribution metric
(KL or token agreement) rather than perplexity.)

## 7. Gemma 4 12B: a dense/SWA model, and why perplexity is the wrong metric

Local checkpoint: Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced, GGUF `Q4_K_M`,
arch `gemma4`, 48 layers = **40 sliding-window layers** (window 1024, 8 KV heads
x 256 dim) + **8 global layers** (1 KV head x 512 dim), 262k vocab.

Its perplexity is unusable as a measurement: the same quantized tiers give
baseline 302 on wikitext, 355 on a human-written chat corpus, 2.24 on text the
model generated itself, and 18.87 on a second self-generated corpus — and in the
last case *quantizing improves* ppl (18.87 -> 8.84 for `q4_0` K+V), because the
noise flattens an over-confident distribution. Heavily finetuned models are
calibrated to their own domain, so ppl measures calibration, not distortion.

Use `llama-perplexity --kl-divergence --save-all-logits` instead (mean KL of the
quantized distribution against the f16 one, plus top-1 agreement). Corpus =
20k tokens generated by the model itself, c=2048 x 4 chunks (8192 scored):

| KV | mean KL | ppl ratio | same top-1 |
|---|---|---|---|
| f16 (sanity) | 0.000000 | 1.006 | 100.0% |
| `q8_0` K+V | 0.1678 | 1.034 | 93.4% |
| `q4_0` K, f16 V | 0.2295 | 1.056 | 90.6% |
| f16 K, `q4_0` V | 0.2247 | 1.123 | 90.8% |
| `q4_0` K+V | 0.3027 | 1.127 | 89.1% |

Reading:

- **No cliff at 12B**, but 4-bit is not free either: KL 0.30 vs 0.17 at 8 bits,
  and top-1 agreement drops to 89%.
- **K and V contribute equally in KL** (0.2295 vs 0.2247) even though ppl calls V
  twice as bad (+12.3% vs +5.6%). ppl and distribution distance disagree about
  which side matters; only the latter is a reliable target.
- The f16 sanity row also fixes the run noise: KL is exactly 0, while the ppl
  ratio drifts by 0.6% between identical runs — another reason to report KL.

For comparison, the same KL protocol on the 27B hybrid (wikitext, c=2048 x 4):

| KV | mean KL | ppl ratio | same top-1 |
|---|---|---|---|
| `q8_0` K+V | 0.0021 | 0.997 | 97.8% |
| `q4_0` K+V | 0.0035 | 1.000 | 96.9% |

So the 27B distorts **~86x less** than the 12B at 4 bits. That is not a
parameter-count effect but an **exposure** effect: in `qwen35` only 17 of 64
layers have a KV cache (the rest are linear attention with recurrent state that
`-ctk/-ctv` never touches), while in `gemma4` every layer's attention reads the
quantized cache. Dense small-to-mid models are therefore the only place where
KV compression is both wanted and measurably costly — and at 12B the cost of
4-bit is a KL of 0.30, not a collapse.

Capacity is still not the motivator here: Gemma4-12B keeps ~32 KiB of KV per
token in its 8 global layers and its 40 SWA layers are capped at the 1024-token
window, so 8k context costs ~0.9 GiB — there is no byte pressure to relieve.

## 8. HF-side KL pre-judgment across model sizes

Before writing anything into llama.cpp, the same A/B was run inside HF
(`kv_probe_kl.py`): 16 windows x 1024 tokens, scored on the last 512, fp16,
wikitext, metric = mean KL(tier || f16) and top-1 agreement. Equal-byte pairs
are adjacent rows.

| model | tier | KL | top-1 | of fp16 |
|---|---|---|---|---|
| Qwen3-8B (36 layers, 8 KV heads, plain dense GQA) | `q8_0` K+V | 0.0009 | 99.2% | 53.1% |
| | dct b32 + q8 K+V | **0.0005** | 99.1% | 53.1% |
| | `q4_0` K+V | 0.0802 | 89.0% | 28.1% |
| | dct b32 + q4 K+V | 0.0852 | 88.4% | 28.1% |
| | `q8_0` K + `q4_0` V | 0.0047 | 97.3% | 40.6% |
| | dct b32 + q8 K, q4 V | 0.0045 | 97.2% | 40.6% |
| | dct b32 + q4 K, q8 V | 0.0541 | 90.5% | 40.6% |
| Qwen3-0.6B (28 layers, 8 KV heads) | `q8_0` K+V | 0.0168 | 93.8% | 53.1% |
| | dct b32 + q8 K+V | **0.0056** | **95.8%** | 53.1% |
| | `q4_0` K+V | 2.3752 | 33.2% | 28.1% |
| | dct b32 + q4 K+V | 1.0283 | 53.0% | 28.1% |
| Hy-MT2-1.8B (32 layers, 4 KV heads) | `q8_0` K+V | 0.0002 | 99.4% | 53.1% |
| | dct b32 + q8 K+V | 0.0003 | 99.2% | 53.1% |
| | `q4_0` K+V | 0.0364 | 90.4% | 28.1% |
| | dct b32 + q4 K+V | 0.0968 | 85.0% | 28.1% |
| Gemma4-E2B (35 layers, 7 full + 28 SWA, 1 KV head) | `q8_0` K+V | 0.0025 | 98.3% | 22.8% |
| | dct b32 + q8 K+V | 0.0028 | 98.1% | 22.8% |
| | `q4_0` K+V | 0.2788 | 79.7% | 12.1% |
| | dct b32 + q4 K+V | 0.3301 | 78.8% | 12.1% |

Qwen3-8B is the decisive row because it is the architecture the codec was
designed against — plain dense GQA, no SWA, no hybrid, no MQA:

- At the operating point that actually matters — `q8_0` K + `q4_0` V, 40.6% of
  fp16 bytes — the transform changes **nothing**: KL 0.0047 raw vs 0.0045 with
  DCT on K, top-1 97.3% vs 97.2%. Spending the 4 bits on V instead of K is worth
  10× more than transforming K.
- At 8 bits both are effectively free (0.0009 vs 0.0005).
- At 4 bits on both sides the transform is **6% worse** (0.0802 -> 0.0852).
- K is again ~18× more sensitive than V (0.0748 vs 0.0042 at `q4_0`), i.e. the
  K/V asymmetry of Qwen3-0.6B is an architecture property, not a size effect.

Across all four models the transform wins only where raw row quantization is
already destructive (0.6B: 3× lower KL at 8 bits, 2.3× at 4 bits — and there
4-bit stays broken), is neutral or slightly better at 8 bits elsewhere, and is
worse at 4 bits except on the model that is already broken.

**Verdict: do not implement this in llama.cpp** (no S2, no kernel). The cheap
pre-judgment answers the question in an afternoon instead of a week, and a plain
dense 8B — the most favourable case for the codec — closes it.

Caveats: 8k–16k scored tokens per model; the harness keeps the full 1024-token
history for SWA layers (no window cropping), so the byte ratios are relative to
an uncropped fp16 cache; the 0.6B/1.8B/Gemma4 checkpoints are finetunes (the
Qwen3-8B one is the base model); the DCT is applied to every layer.

## 9. What this means for flashcodec

- The seq-axis transform is a **real distortion win at equal bytes**, but only
  where row quantization is already destructive: 3× lower KL at 8 bits and 2.3×
  at 4 bits on Qwen3-0.6B, and **neutral at 8 bits / worse at 4 bits** on
  1.8–2B models that quantize cleanly (§8). Combined with 4-bit still being
  broken at 0.6B (KL 1.03, top-1 53%), there is no operating point where the
  codec is worth shipping: **no llama.cpp implementation follows from this.**
- What the exercise did produce is a set of transferable measurements:
  the coefficient quantizer, not the transform, decides the earlier verdict
  (§5); the error is redistributed rather than reduced (§5); perplexity and KL
  disagree about which side (K or V) matters (§7); and KV-quantization damage is
  driven by architectural exposure more than by parameter count (§6–§7).
- **Metric discipline**: use KL / top-1 agreement (`--kl-divergence`) for
  finetuned or dense models — their ppl is a calibration artifact and sometimes
  *improves* under quantization. ppl is only informative on base models with a
  sane baseline (the 0.6B and 27B rows).
- Deployment advice is unchanged and now has a calibration-free backing:
  `-ctk q8_0 -ctv q4_0` on base models (40.6% of fp16 KV), never `-ctk q4_0` on
  a small dense model (KL 2.4, top-1 33% at 0.6B).

## 10. Limitations

- One model (Qwen3-0.6B), one context length (512). Longer contexts usually make
  KV quantization *more* damaging, but they should not reorder DCT vs row
  quantization, because the two have different error mechanisms.
- HF bf16 vs llama.cpp fp16; `q8_0 emul` is a semantic emulation, not bit-exact.
- Hybrid models (Qwen3.5-0.8B) are not re-measured here: only 6 of 24 layers
  carry a codec there, so the absolute damage is small, but at equal bytes those
  six layers are still better served by `q8_0`.
- Capacity: at 0.6B and 128k context, `K=q8_0 + V=q4_0` needs only ~4.7 GB, so
  there is no VRAM pressure on a 32 GB card. The capacity argument only bites
  for large models at very long context — where `q8_0` K + `q4_0` V is still the
  cheaper answer.

## 11. Reproduce

```bash
# convert (needs the `gguf` + `sentencepiece` python packages)
python llama.cpp/convert_hf_to_gguf.py <qwen3-0.6b-hf-dir> \
    --outfile qwen3-0.6b-f16.gguf --outtype f16

# llama.cpp matrix
llama.cpp/build/bin/llama-perplexity -m qwen3-0.6b-f16.gguf \
    -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 40 \
    -ngl 99 -fa on -ctk q8_0 -ctv q4_0

llama.cpp/build/bin/llama-bench -m qwen3-0.6b-f16.gguf \
    -p 4096 -n 32 -fa 1 -ngl 99 -ctk q8_0 -ctv q8_0 -r 3
```

The HF-side A/B harness (K-only / V in {raw, quant, codec}, ggml-style row
quantization emulation, DCT tiers) is a standalone script kept with the
experiment record; it is not part of the published package.

## Appendix: why this is about KV and not about model weights

The original idea was JPEG-for-weights: keep only the low-frequency
coefficients and shrink the model by 75%. That needs an autocorrelated axis,
and weights do not have one — an `(out, in)` matrix has no sequence axis, and
its trained spectrum is flat (high stable rank), so "keep the low-frequency
quarter" degenerates into a low-rank truncation that costs far more than 75% of
the information. The same asymmetry was measured here on KV: truncation needs
**2.7× the bytes** of block quantization for the same damage (§2).

The workable form of the same intuition is not truncation but **rotate, then
quantize**: Hadamard / random rotations spread the energy before per-block
quantization (QuaRot, SpinQuant; llama.cpp ships this as `hdmd`), and hybrid
low-rank-plus-quantization keeps the dominant subspace while encoding the rest
at a lower bit width (SVD-LLM, ASVD). If the weights route is ever revisited,
the cheapest test is one afternoon on a local dense model — at a fixed 25% of
fp16 bytes, compare (a) per-matrix 2-D DCT truncation, (b) Hadamard +
per-block 4-bit, (c) SVD rank-r, on perplexity.

### Two conditions the JPEG analogy needs

Transform coding only pays off when (a) the transform concentrates energy and
(b) an *autocorrelated axis that carries the damage* exists at all. Both are
measurable, and text KV fails them differently from image patches:

- **Energy concentration**: for a first-order process `Var(Δ)/Var(X) = 2(1−ρ)`.
  The measured text-KV statistics map to ρ ≈ 0.64–0.92 for K (smooth — the
  transform does work, and it is the side that benefits in every run above) and
  ρ ≈ −0.07–0.82 for V (often anti-correlated — the transform cannot help).
- **A naive baseline quantizer**: JPEG beats *uniform* quantization because its
  table adds frequency adaptation. ggml's per-32 absmax is already an adaptive
  table, so most of that gain is spent before the transform arrives — this is
  exactly the 0.6B / 1.8B / 8B spread in §8.
- **The damage is not in the frequency domain**: K is the sensitive side because
  of outlier channels along `head_dim`, not because of its spectrum. So the
  effective "signal processing before quantization" for K is a **rotation along
  `head_dim`** (QuaRot, SpinQuant; shipped in llama.cpp as `hdmd`) — the same
  architecture of idea as this project, applied to the axis where the error
  actually lives.
