"""Perplexity cost of the narrower coefficient tiers, with and without the
calibrated channel profile.

The point of the profile is that the fp32 scale is shared by all head_dim
channels; calibrating a per-(frequency, channel) table is what makes 7-bit
(and, more marginally, 6-bit) usable at all. The numbers below are the ones in
the README.

Corpus: by default the frozen `docs/eval-corpus.txt` (a snapshot of the
repository's own markdown: README.zh-CN.md, README.md, AGENTS.md, CHANGELOG.md,
docs/*.md). It is frozen on purpose — building the windows from the live docs
would make every README edit move the measured numbers. Set
FLASHCODEC_CORPUS to a JSONL file with a "text" field to use your own data (the
calibration slice is taken after the evaluated windows, so it never overlaps
them).

Usage:
    python benchmarks/bench_calibrated.py
Environment:
    FLASHCODEC_MODEL        model id / path (default Qwen/Qwen3-0.6B, offline)
    FLASHCODEC_CORPUS       optional JSONL corpus
    FLASHCODEC_PPL_WINDOWS  number of eval windows (default 16)
    FLASHCODEC_PPL_SEQ      window length (default 512)
    FLASHCODEC_CALIB_TOKENS calibration tokens (default 2048)
"""

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashcodec import FlashCodecCache, calibrate_channel_scales

MODEL = os.environ.get("FLASHCODEC_MODEL", "Qwen/Qwen3-0.6B")
N = int(os.environ.get("FLASHCODEC_PPL_WINDOWS", "16"))
SEQ = int(os.environ.get("FLASHCODEC_PPL_SEQ", "512"))
CAL = int(os.environ.get("FLASHCODEC_CALIB_TOKENS", "2048"))
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"model={MODEL}  device={device}  windows={N} x {SEQ}  calib={CAL} tokens\n")
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True,
    dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()

corpus = os.environ.get("FLASHCODEC_CORPUS")
frozen = ROOT / "docs" / "eval-corpus.txt"
if corpus:
    texts = []
    with open(corpus, encoding="utf-8") as fh:
        for line in fh:
            try:
                texts.append(json.loads(line).get("text", ""))
            except json.JSONDecodeError:
                pass
elif frozen.exists():
    texts = [frozen.read_text(encoding="utf-8")]
else:
    order = ["README.zh-CN.md", "README.md", "AGENTS.md", "CHANGELOG.md"]
    paths = [ROOT / n for n in order] + sorted((ROOT / "docs").glob("*.md"))
    texts = [p.read_text(encoding="utf-8") for p in paths if p.exists()]

ids = []
for t in texts:
    if t:
        ids.extend(tok(t, add_special_tokens=False).input_ids)
print(f"corpus tokens: {len(ids)}")
need = N * SEQ + CAL
if len(ids) < need:
    sys.exit(f"corpus too small: need {need} tokens, have {len(ids)}")

windows = [torch.tensor(ids[i * SEQ:(i + 1) * SEQ], device=device).unsqueeze(0)
           for i in range(N)]
calib_ids = torch.tensor(ids[N * SEQ:N * SEQ + CAL], device=device).unsqueeze(0)
calib = calibrate_channel_scales(model, calib_ids, keep_freqs=8)
print(f"calibrated layers: {len(calib)}  (eval and calibration slices disjoint)\n")

TIERS = [
    ("baseline", None),
    ("k=8 b=8", dict(mode="jpeg", keep_freqs=8, quant_bits=8)),
    ("k=8 b=8 +profile", dict(mode="jpeg", keep_freqs=8, quant_bits=8,
                              per_layer=calib)),
    ("k=8 b=7", dict(mode="jpeg", keep_freqs=8, quant_bits=7)),
    ("k=8 b=7 +profile", dict(mode="jpeg", keep_freqs=8, quant_bits=7,
                              per_layer=calib)),
    ("k=8 b=6", dict(mode="jpeg", keep_freqs=8, quant_bits=6)),
    ("k=8 b=6 +profile", dict(mode="jpeg", keep_freqs=8, quant_bits=6,
                              per_layer=calib)),
]


def eval_ppl(cfg):
    nll = ntok = 0
    for ids_w in windows:
        cache = (FlashCodecCache.from_config(model.config, block_size=8, **cfg)
                 if cfg else None)
        with torch.no_grad():
            out = model(input_ids=ids_w, labels=ids_w,
                        use_cache=cache is not None, past_key_values=cache)
        nll += out.loss.item() * ids_w.shape[1]
        ntok += ids_w.shape[1]
    return math.exp(nll / ntok)


base = None
print(f"{'tier':<20} {'ppl':>10} {'dppl':>9} {'storage':>9}")
print("-" * 52)
for name, cfg in TIERS:
    ppl = eval_ppl(cfg)
    if base is None:
        base = ppl
        ratio = "-"
    else:
        k, b = cfg["keep_freqs"], cfg["quant_bits"]
        ratio = f"{(k * 128 * b / 8 + k * 4) / (8 * 128 * 2) * 100:.1f}%"
    print(f"{name:<20} {ppl:>10.4f} {ppl - base:>+9.4f} {ratio:>9}")

print("\nThe profile is a per-(layer, stream) constant; without it 7-bit and "
      "6-bit fall apart (README 4.1).")
