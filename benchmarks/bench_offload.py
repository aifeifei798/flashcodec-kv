"""Is offloading V to host RAM / a video stream worth it?

Three measurements decide it, all on the real model:
  1. H2D bandwidth (pinned / pageable) with the PCIe link warmed up — GPUs idle
     at Gen2 and only ramp under sustained traffic, so a cold one-shot number
     lies.
  2. the decode step at each context length (the thing the transfer must hide
     behind). It is launch-bound, so it grows much slower than S.
  3. the step with a V-sized H2D stream running concurrently on a side stream:
     that is the best case for any offload (the copy engine is independent of
     the SMs), so `max(step, transfer)` is the floor.

The V volume per step is `S * H_kv * D * n_layers` elements; at 8 bits that is
28.7 KB per token.  If that transfer is bigger than the step, no amount of
pipelining saves it.

Usage:
    python benchmarks/bench_offload.py
    FLASHCODEC_OFFLOAD_CTX=4096,16384 python benchmarks/bench_offload.py
Environment:
    FLASHCODEC_MODEL          model id / path (default Qwen/Qwen3-0.6B)
    FLASHCODEC_OFFLOAD_CTX    comma-separated context lengths
                              (default 4096,16384,65536; 65536 needs ~8 GB VRAM)
    FLASHCODEC_OFFLOAD_STEPS  timed decode steps per context (default 5)
"""

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

MODEL = os.environ.get("FLASHCODEC_MODEL", "Qwen/Qwen3-0.6B")
CTX = [int(x) for x in os.environ.get("FLASHCODEC_OFFLOAD_CTX",
                                      "4096,16384,65536").split(",")]
STEPS = int(os.environ.get("FLASHCODEC_OFFLOAD_STEPS", "5"))
device = "cuda" if torch.cuda.is_available() else "cpu"


def pcie_link():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=pcie.link.gen.current,"
             "pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        gen, gmax, wid, wmax = [x.strip() for x in out.stdout.strip().split(",")]
        return f"Gen{gen} x{wid} (capable Gen{gmax} x{wmax})"
    except Exception:
        return "unknown"


def h2d_bandwidth(nbytes=200_000_000, warm_s=2.0):
    """Pinned / pageable H2D after warming the link up (idle GPUs sit at Gen2)."""
    res = {}
    for pinned in (True, False):
        src = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pinned).fill_(1)
        dst = torch.empty(nbytes, dtype=torch.uint8, device=device)
        t0 = time.time()
        while time.time() - t0 < warm_s:
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 3
        res["pinned" if pinned else "pageable"] = nbytes / dt / 1e9
        del src, dst
    return res


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device).eval()
    return tok, model


def prefill(tok, model, ctx):
    ids = tok(("The quick brown fox jumps over the lazy dog. " * (ctx // 9 + 2)),
              return_tensors="pt", truncation=True,
              max_length=ctx).input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    kv_bytes = sum(l.keys.numel() * l.keys.element_size()
                   + l.values.numel() * l.values.element_size()
                   for l in out.past_key_values.layers)
    return out, kv_bytes


def decode_step(model, out, timed=STEPS):
    pkv, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    with torch.no_grad():
        o = model(input_ids=nxt, past_key_values=pkv, use_cache=True,
                  logits_to_keep=1)                       # warmup
        pkv, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(timed):
            o = model(input_ids=nxt, past_key_values=pkv, use_cache=True,
                      logits_to_keep=1)
            pkv, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / timed
    return dt, pkv, nxt


def overlap_step(model, pkv, nxt, nbytes, chunks=28):
    """Decode step while a V-sized H2D stream runs on a side stream.

    The copy engine is independent of the SMs, so this is the *floor* for any
    host-offload scheme: max(step, transfer)."""
    src = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True).fill_(1)
    dst = torch.empty(nbytes, dtype=torch.uint8, device=device)
    step = (nbytes + chunks - 1) // chunks
    side = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    side.wait_stream(main)
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(side):
            for i in range(chunks):
                a = i * step
                dst[a:min(nbytes, a + step)].copy_(
                    src[a:min(nbytes, a + step)], non_blocking=True)
        o = model(input_ids=nxt, past_key_values=pkv, use_cache=True,
                  logits_to_keep=1)
        main.wait_stream(side)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    del src, dst
    return dt, o.past_key_values, o.logits[:, -1:].argmax(-1)


def nvdec_gpixs():
    """Measured hardware-JPEG decode rate (torchvision/nvJPEG), if available."""
    try:
        from torchvision.io import decode_jpeg, encode_jpeg
        t = torch.arange(4096).float()
        img = (128 + 20 * torch.sin(t[None, :] / 700.0)
               + torch.randn(1024, 4096) * 3).clamp(0, 255).to(torch.uint8)[None]
        b = encode_jpeg(img, quality=90)
        for _ in range(2):
            decode_jpeg(b, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            decode_jpeg(b, device="cuda")
        torch.cuda.synchronize()
        return img.numel() / ((time.perf_counter() - t0) / 5) / 1e9
    except Exception as e:                    # no torchvision / no nvJPEG
        print(f"   (nvJPEG unavailable: {type(e).__name__}: {e})")
        return None


def kv_split(model, tok, bits=(7, 6), n_win=16, seq=512, calib_tokens=2048):
    """Is V really the tolerant stream? Compress K only / V only / both.

    The codec swap happens before the first forward, when the archive is empty,
    so it only changes how the stream *will* be compressed ("lossless" = full
    DCT basis, no quantization)."""
    from flashcodec import FlashCodecCache, JPEGCodec, calibrate_channel_scales
    order = ["README.zh-CN.md", "README.md", "AGENTS.md", "CHANGELOG.md"]
    ids = []
    for name in order:
        p = ROOT / name
        if p.exists():
            ids.extend(tok(p.read_text(encoding="utf-8"),
                           add_special_tokens=False).input_ids)
    if len(ids) < n_win * seq + calib_tokens:
        print("   (kv-split skipped: repo markdown too short)")
        return
    windows = [torch.tensor(ids[i * seq:(i + 1) * seq], device=device).unsqueeze(0)
               for i in range(n_win)]
    calib = calibrate_channel_scales(
        model, torch.tensor(ids[n_win * seq:n_win * seq + calib_tokens],
                            device=device).unsqueeze(0), keep_freqs=8)
    loss = JPEGCodec(8, 8, quant_bits=0)

    def run(keep, bits):
        nll = ntok = 0
        for w in windows:
            cache = None
            if keep:
                cache = FlashCodecCache.from_config(
                    model.config, block_size=8, mode="jpeg", keep_freqs=8,
                    quant_bits=bits, per_layer=calib)
                for layer in cache.layers:
                    if not hasattr(layer, "_key_stream"):
                        continue
                    if keep in ("v", "both"):
                        layer._key_stream.codec = loss
                        layer.codec = loss
                    if keep in ("k", "both"):
                        layer._value_stream.codec = loss
                        layer.value_codec = loss
            with torch.no_grad():
                out = model(input_ids=w, labels=w, use_cache=cache is not None,
                            past_key_values=cache)
            nll += out.loss.item() * w.shape[1]
            ntok += w.shape[1]
        return math.exp(nll / ntok)

    base = run(None, 0)
    print(f"\nK/V split (calibrated profile, {n_win} x {seq} windows, "
          f"baseline {base:.4f})")
    print(f"{'bits':>4} {'K+V':>10} {'V only':>10} {'K only':>10}")
    for b in bits:
        both, vonly, konly = run("both", b), run("v", b), run("k", b)
        print(f"{b:4d} {both-base:+9.4f} {vonly-base:+9.4f} {konly-base:+9.4f}")
    print("   (K-only means V is lossless; V-only means K is lossless)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-split", action="store_true",
                    help="also run the K-only / V-only perplexity comparison")
    args = ap.parse_args()
    if device != "cuda":
        sys.exit("bench_offload needs a CUDA device")
    print(f"model={MODEL}  device={torch.cuda.get_device_name(0)}\n")
    print(f"PCIe link (current, idle): {pcie_link()}")
    bw = h2d_bandwidth()
    print(f"H2D bandwidth (warmed): pinned {bw['pinned']:.1f} GB/s, "
          f"pageable {bw['pageable']:.1f} GB/s\n")

    tok, model = load_model()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    # every layer carries a codec on a dense decoder; hybrid models keep only
    # full_attention layers, which is what makes the per-token volume smaller
    codec_layers = sum(1 for t in getattr(cfg, "layer_types", ["full_attention"] * n_layers)
                       if t == "full_attention")
    v_bytes_per_tok = codec_layers * kv_heads * head_dim     # at 8 bits

    gpixs = nvdec_gpixs()
    print(f"K/V per token: {v_bytes_per_tok/1e3:.1f} KB (V at 8 bits, "
          f"{codec_layers} codec layers of {n_layers})\n")

    rows = []
    for ctx in CTX:
        try:
            out, kv_bytes = prefill(tok, model, ctx)
        except torch.cuda.OutOfMemoryError:
            print(f"ctx={ctx}: skipped (OOM)")
            continue
        step, pkv, nxt = decode_step(model, out)
        vbytes = v_bytes_per_tok * ctx
        tfer = vbytes / bw["pinned"] / 1e9
        ov, _, _ = overlap_step(model, pkv, nxt, vbytes)
        rows.append((ctx, kv_bytes, step, vbytes, tfer, ov))
        print(f"ctx={ctx:6d}  KV={kv_bytes/1e9:.2f}GB  step={step*1e3:6.1f}ms  "
              f"V={vbytes/1e6:7.0f}MB  transfer={tfer*1e3:6.1f}ms  "
              f"concurrent step={ov*1e3:6.1f}ms  "
              f"({ov/step:.1f}x)")
        del out, pkv
        torch.cuda.empty_cache()

    print(f"\n{'ctx':>6} {'step':>8} {'V (8bit)':>9} {'xfer':>8} "
          f"{'xfer/step':>10} {'concurrent':>11} {'vs step':>8} "
          f"{'nvJPEG/step':>11}")
    for ctx, kv_bytes, step, vbytes, tfer, ov in rows:
        npix = codec_layers * ctx * kv_heads * head_dim
        nv = (npix / gpixs / 1e9) if gpixs else float("nan")
        print(f"{ctx:6d} {step*1e3:6.1f}ms {vbytes/1e6:7.0f}MB {tfer*1e3:6.1f}ms "
              f"{tfer/step:9.1f}x {ov*1e3:9.1f}ms {ov/step:7.1f}x "
              f"{nv*1e3:9.1f}ms")

    if args.kv_split:
        kv_split(model, tok)
        torch.cuda.empty_cache()

    print("\nread: 'concurrent' is the decode step with the V transfer running "
          "on a side stream —\n      the hardware floor for any host-offload "
          "scheme (copy engine is independent of the SMs).")
    print("      'nvJPEG/step' decodes V through NVDEC every step, as in the "
          "'V as a video stream' proposal;\n      doing it once at archive "
          "time is fine, doing it per step is not.")


if __name__ == "__main__":
    main()
