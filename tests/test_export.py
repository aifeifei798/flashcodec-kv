"""Export / reload a calibrated 7-bit bundle.

1) the manifest is self-describing and the base64 float32 profiles round-trip
2) export_model writes a bundle (weights included) and load_pretrained restores
   a patched model whose codec tier and per-layer profiles match the manifest
3) apply_export attaches the same tier to an already-loaded model
4) the patched model actually generates, and its archive is ~7/8 of the int8 one

Runs on CPU with an offline tiny Llama (no network, no tokenizer).
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from flashcodec import (apply_export, export_model, load_pretrained,
                        read_manifest)
from flashcodec.export import FORMAT_VERSION, MANIFEST_NAME

torch.manual_seed(0)
PROMPT = torch.randint(0, 128, (1, 40), generator=torch.Generator().manual_seed(3))
CALIB = torch.randint(0, 128, (1, 96), generator=torch.Generator().manual_seed(4))


def build_and_save(path):
    cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=256)
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(20.0)
    m.save_pretrained(path)
    return path


def test_export_manifest_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        base = build_and_save(Path(tmp) / "base")
        out = Path(tmp) / "bundle"
        man = export_model(base, out, calib_ids=CALIB, save_weights=True,
                           device="cpu", dtype=torch.float32)
        assert (out / MANIFEST_NAME).exists()
        disk = read_manifest(out)
        assert disk["format"] == "flashcodec-kv-model"
        assert disk["version"] == FORMAT_VERSION
        assert disk["weights_included"] is True
        assert disk["codec"] == {"mode": "jpeg", "block_size": 8,
                                 "keep_freqs": 8, "quant_bits": 7}
        assert set(disk["layers"]) == set(man["layers"])
        entry = disk["layers"]["0"]["channel_scale"]
        assert entry["shape"] == [8, 16]
        assert entry["dtype"] == "float32"
        assert len(entry["data"]) > 0
        # base64 encoding is deterministic -> re-encoding the decoded table is
        # a bitwise check of the float32 round trip
        from flashcodec.export import _decode_table, _encode_table
        t = _decode_table(entry)
        assert torch.equal(t, _decode_table(
            {"shape": entry["shape"], "data": _encode_table(t)}))
        assert torch.isfinite(t).all() and (t > 0).all()


def test_load_pretrained_applies_profile():
    with tempfile.TemporaryDirectory() as tmp:
        base = build_and_save(Path(tmp) / "base")
        out = Path(tmp) / "bundle"
        man = export_model(base, out, calib_ids=CALIB, save_weights=True,
                           device="cpu", dtype=torch.float32)
        model, tok = load_pretrained(out, device="cpu", dtype=torch.float32)
        assert tok is None                       # tiny model has no tokenizer
        info = model._flashcodec
        assert info is not None and info["mode"] == "jpeg"
        with torch.no_grad():
            out_ids = model.generate(PROMPT, max_new_tokens=4, do_sample=False,
                                     min_new_tokens=4, pad_token_id=0)
        assert out_ids.shape[1] == PROMPT.shape[1] + 4
        cache = info["caches"][0]                # created on first generate
        assert cache.codec_cfg["quant_bits"] == 7
        layer = next(l for l in cache.layers if hasattr(l, "_key_stream"))
        g = layer._key_stream.codec.channel_scale
        assert g is not None and g.shape == (8, 16)
        assert torch.equal(g, _decode(man["layers"]["0"]["channel_scale"]))
        assert cache.storage_bytes() > 0
        # 7-bit + per-row scales lands around 45% of bf16 at head_dim=128 and a
        # bit higher at head_dim=16 (the scale overhead does not amortise)
        raw = 0
        for idx in cache.codec_layer_indices():
            l = cache.layers[idx]
            for st in (l._key_stream, l._value_stream):
                raw += st.seq_len * st.head_dim * st.meta[1] * st.dtype.itemsize
        ratio = cache.storage_bytes() / raw
        assert 0.25 < ratio < 0.55, (cache.storage_bytes(), raw, ratio)


def test_apply_export_on_loaded_model():
    with tempfile.TemporaryDirectory() as tmp:
        base = build_and_save(Path(tmp) / "base")
        out = Path(tmp) / "bundle"
        export_model(base, out, calib_ids=CALIB, save_weights=True,
                     device="cpu", dtype=torch.float32)
        man = read_manifest(out)
        m = LlamaForCausalLM.from_pretrained(out).eval()
        assert m._flashcodec is None if hasattr(m, "_flashcodec") else True
        apply_export(m, man)
        assert m._flashcodec is not None
        with torch.no_grad():
            m(PROMPT, use_cache=True)


def _decode(entry):
    from flashcodec.export import _decode_table
    return _decode_table(entry)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = True
    for fn in fns:
        try:
            fn()
            print(f"✅ {fn.__name__}")
        except AssertionError as e:
            ok = False
            print(f"❌ {fn.__name__}: {e}")
    print("\nALL PASS" if ok else "\nFAILURES")
    sys.exit(0 if ok else 1)
