"""Export an HF checkpoint + a calibrated 7-bit KV profile into one bundle.

What "export a 7-bit model" means here
--------------------------------------
flashcodec compresses the **KV cache at inference time**; the weights are never
touched.  So the artifact is the original checkpoint plus a
``flashcodec-kv.json`` sidecar carrying:

- the codec tier (``mode="jpeg"``, ``keep_freqs``, ``quant_bits``, ``block_size``),
- the per-layer, per-stream channel profiles produced by
  ``calibrate_channel_scales`` (base64 float32, a few hundred KB),
- the calibration provenance (token count, where the text came from).

Loading applies both to the model in one call, so a user never has to know how
the profile is wired.  Without the profile, 7-bit is not usable (see README
4.1: +2.34 ppl vs +0.03 with it).

CLI::

    python -m flashcodec.export Qwen/Qwen3-0.6B ./qwen3-0.6b-fckv \\
        --calib-text calibration.txt
    python -m flashcodec.export --inspect ./qwen3-0.6b-fckv

Then::

    from flashcodec import load_pretrained
    model, tok = load_pretrained("./qwen3-0.6b-fckv")
"""

import argparse
import base64
import json
import struct
import sys
import time
from pathlib import Path

import torch

from .calibrate import calibrate_channel_scales
from .patch import apply_flash_codec

__all__ = ["export_model", "load_pretrained", "read_manifest", "apply_export",
           "MANIFEST_NAME", "FORMAT_VERSION"]

MANIFEST_NAME = "flashcodec-kv.json"
FORMAT_VERSION = 1

# Used when the caller does not pass its own calibration text.  A few hundred
# tokens of mixed prose is enough to estimate per-channel magnitudes; pass
# `--calib-text` with in-domain text for the best numbers.
_BUILTIN_CALIB = (
    "KV cache compression trades bytes for attention fidelity. The keys carry "
    "the matching signal, the values carry the retrieved content, and both are "
    "produced by the same projection matrices in every layer. A calibrated "
    "per-channel profile removes the mismatch between a single scale shared by "
    "128 channels and the per-channel magnitudes of post-RoPE keys. "
    "大模型推理时，KV cache 会随上下文线性增长，长上下文场景下显存成为瓶颈。"
    "把键值缓存按块做正交变换、量化并配合通道校准，可以在几乎不损失困惑度的"
    "情况下把缓存压到原来的一半以下。"
    "In practice the decode step is memory bound, so shrinking the cache helps "
    "both capacity and bandwidth; prefix caching, speculative decoding and "
    "paged attention all interact with how the cache is laid out in memory. "
    "The quick brown fox jumps over the lazy dog. 0123456789 "
    "def attention(q, k, v): return softmax(q @ k.T) @ v  # reference form\n"
)


# ------------------------------------------------------------- profile codec ---

def _encode_table(t):
    """fp32 tensor -> base64 of little-endian float32 (numpy-free)."""
    t = t.detach().to(torch.float32).cpu().contiguous().reshape(-1)
    return base64.b64encode(
        struct.pack("<%df" % t.numel(), *t.tolist())).decode("ascii")


def _decode_table(entry):
    raw = base64.b64decode(entry["data"])
    vals = struct.unpack("<%df" % (len(raw) // 4), raw)
    return torch.tensor(vals, dtype=torch.float32).reshape(*entry["shape"])


def _table_entry(t):
    return {"shape": list(t.shape), "dtype": "float32", "data": _encode_table(t)}


# ---------------------------------------------------------------- calibration ---

def _calib_ids(tokenizer, texts, n_tokens):
    if tokenizer is None:
        raise ValueError("a tokenizer is required when calibrating from text")
    ids = []
    for text in texts:
        ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
        if len(ids) >= n_tokens:
            break
    if len(ids) < 64:
        raise ValueError(f"calibration text is too short ({len(ids)} tokens); "
                         f"pass --calib-text with a few thousand tokens")
    return torch.tensor(ids[:n_tokens]).unsqueeze(0)


# --------------------------------------------------------------------- export ---

def export_model(model_id_or_path, out_dir, *, keep_freqs=8, quant_bits=7,
                 block_size=8, calib_texts=None, calib_ids=None,
                 calib_tokens=2048, quantile=0.95, save_weights=False,
                 dtype=torch.bfloat16, device="cuda", overwrite=False,
                 local_files_only=False, trust_remote_code=False):
    """Calibrate ``model_id_or_path`` and write a bundle to ``out_dir``.

    ``save_weights=True`` writes a self-contained directory (model +
    tokenizer + manifest); otherwise the manifest points at
    ``model_id_or_path`` and only the profiles are stored.

    Returns the manifest dict (also written to ``out_dir/flashcodec-kv.json``).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} is not empty (pass overwrite=True)")
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id_or_path} ...", flush=True)
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path, local_files_only=local_files_only,
            trust_remote_code=trust_remote_code)
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path, local_files_only=local_files_only,
        trust_remote_code=trust_remote_code, dtype=dtype).to(device).eval()

    maxpos = getattr(model.config, "max_position_embeddings", None)
    if maxpos and calib_tokens > int(maxpos):
        print(f"clamping calibration to the model's {maxpos} positions")
        calib_tokens = int(maxpos)
    texts = list(calib_texts) if calib_texts else [_BUILTIN_CALIB]
    if not calib_texts:
        print("no --calib-text: calibrating on a small builtin sample "
              "(pass in-domain text for the best profile)")
    if calib_ids is None:
        calib_ids = _calib_ids(tokenizer, texts, calib_tokens)
    calib_ids = calib_ids.to(device)
    print(f"calibrating on {calib_ids.shape[1]} tokens "
          f"(keep_freqs={keep_freqs}, quant_bits={quant_bits}) ...", flush=True)
    t0 = time.time()
    scales = calibrate_channel_scales(model, calib_ids, block_size=block_size,
                                      keep_freqs=keep_freqs, quantile=quantile)
    print(f"  {len(scales)} layers in {time.time() - t0:.1f}s", flush=True)
    if not scales:
        raise RuntimeError(
            f"{model_id_or_path} exposed no KV-cache layers to calibrate "
            "(not a standard decoder-only CausalLM, or its cache is not a "
            "transformers 5.x Cache with .layers)")

    manifest = {
        "format": "flashcodec-kv-model",
        "version": FORMAT_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": str(model_id_or_path),
        "weights_included": bool(save_weights),
        "codec": {"mode": "jpeg", "block_size": block_size,
                  "keep_freqs": keep_freqs, "quant_bits": quant_bits},
        "calibration": {"tokens": int(calib_ids.shape[1]),
                        "quantile": quantile,
                        "source": "user text" if calib_texts else "builtin"},
        "layers": {},
    }
    for idx, d in scales.items():
        entry = {}
        for src, dst in (("channel_scale", "channel_scale"),
                         ("channel_scale_v", "channel_scale_v")):
            if d.get(src) is not None:
                entry[dst] = _table_entry(d[src])
        manifest["layers"][str(idx)] = entry

    if save_weights:
        print(f"saving weights + tokenizer to {out} ...", flush=True)
        model.save_pretrained(out)
        if tokenizer is not None:
            tokenizer.save_pretrained(out)
        manifest["base_model"] = str(out.resolve())

    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1) + "\n",
                                     encoding="utf-8")
    nbytes = (out / MANIFEST_NAME).stat().st_size
    print(f"wrote {out / MANIFEST_NAME} ({nbytes/1e3:.0f} KB, "
          f"{len(manifest['layers'])} layers)")
    return manifest


# --------------------------------------------------------------------- load ---

def read_manifest(path):
    """Read a manifest from a bundle directory or a ``flashcodec-kv.json``."""
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST_NAME
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist")
    manifest = json.loads(p.read_text(encoding="utf-8"))
    if manifest.get("format") != "flashcodec-kv-model":
        raise ValueError(f"{p}: not a flashcodec model manifest")
    if manifest.get("version", 0) > FORMAT_VERSION:
        raise ValueError(f"{p}: manifest version "
                         f"{manifest['version']} is newer than this package "
                         f"({FORMAT_VERSION})")
    return manifest


def _per_layer(manifest):
    out = {}
    for idx, entry in manifest["layers"].items():
        d = {}
        for name in ("channel_scale", "channel_scale_v"):
            if name in entry:
                d[name] = _decode_table(entry[name])
        out[int(idx)] = d
    return out


def apply_export(model, manifest, materialization_free=False):
    """Attach the manifest's codec tier + profiles to an already-loaded model."""
    cfg = dict(manifest["codec"])
    return apply_flash_codec(model, mode=cfg["mode"],
                             block_size=cfg["block_size"],
                             keep_freqs=cfg["keep_freqs"],
                             quant_bits=cfg["quant_bits"],
                             per_layer=_per_layer(manifest),
                             materialization_free=materialization_free)


def load_pretrained(path, device=None, dtype=None, materialization_free=False,
                    local_files_only=False, trust_remote_code=False,
                    **model_kwargs):
    """Load an exported bundle (or plain model dir + sidecar) and patch it.

    Returns ``(model, tokenizer)`` — ``tokenizer`` is ``None`` when the bundle
    does not carry one.  The cache codec (7-bit by default) is active for
    ``generate`` / ``forward`` immediately.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = read_manifest(path)
    base = manifest["base_model"]
    model_dir = Path(path)
    if model_dir.is_dir() and manifest.get("weights_included"):
        base = str(model_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base, local_files_only=local_files_only,
        trust_remote_code=trust_remote_code, dtype=dtype,
        **model_kwargs).to(device)
    model.eval()
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base, local_files_only=local_files_only,
            trust_remote_code=trust_remote_code)
    except Exception:
        pass
    apply_export(model, manifest, materialization_free=materialization_free)
    return model, tokenizer


# ---------------------------------------------------------------------- CLI ---

def _inspect(path):
    m = read_manifest(path)
    c = m["codec"]
    print(f"format   : {m['format']} v{m['version']}  ({m['created']})")
    print(f"base     : {m['base_model']}  "
          f"(weights {'included' if m['weights_included'] else 'referenced'})")
    print(f"codec    : {c['mode']} block={c['block_size']} "
          f"keep_freqs={c['keep_freqs']} quant_bits={c['quant_bits']}")
    print(f"calib    : {m['calibration']['tokens']} tokens "
          f"({m['calibration']['source']}), q{m['calibration']['quantile']}")
    print(f"layers   : {len(m['layers'])} "
          f"({', '.join(sorted(m['layers'], key=int)[:8])}"
          f"{' ...' if len(m['layers']) > 8 else ''})")
    sizes = sorted({tuple(e['channel_scale']['shape'])
                    for e in m['layers'].values() if 'channel_scale' in e})
    print(f"profiles : {sizes}")
    k, b = c['keep_freqs'], c['quant_bits']
    dims = [s[1] for s in sizes] or [128]
    for D in dims:
        print(f"storage  : ~{(k*D*b/8 + k*4)/(c['block_size']*D*2)*100:.1f}% "
              f"of bf16 KV (head_dim={D}, one stream)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m flashcodec.export",
        description="Export an HF model together with a calibrated 7-bit KV "
                    "profile (weights are untouched).")
    ap.add_argument("model", nargs="?", help="HF id or local checkpoint path")
    ap.add_argument("out", nargs="?", help="output bundle directory")
    ap.add_argument("--inspect", metavar="BUNDLE",
                    help="print a bundle's manifest and exit")
    ap.add_argument("--quant-bits", type=int, default=7)
    ap.add_argument("--keep-freqs", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--calib-text", action="append", default=None,
                    help="text file used for calibration (repeatable)")
    ap.add_argument("--calib-tokens", type=int, default=2048)
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--save-weights", action="store_true",
                    help="copy weights + tokenizer into the bundle")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="only use locally cached model files")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="needed by some custom architectures")
    args = ap.parse_args(argv)

    if args.inspect:
        _inspect(args.inspect)
        return 0
    if not args.model or not args.out:
        ap.error("MODEL and OUT are required unless --inspect is used")
    texts = None
    if args.calib_text:
        texts = [Path(p).read_text(encoding="utf-8") for p in args.calib_text]
    export_model(args.model, args.out,
                 keep_freqs=args.keep_freqs, quant_bits=args.quant_bits,
                 block_size=args.block_size, calib_texts=texts,
                 calib_tokens=args.calib_tokens, quantile=args.quantile,
                 save_weights=args.save_weights,
                 dtype=getattr(torch, args.dtype), device=args.device,
                 overwrite=args.overwrite, local_files_only=args.offline)
    print(f"\nload it with:\n  from flashcodec import load_pretrained\n"
          f"  model, tok = load_pretrained({args.out!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
