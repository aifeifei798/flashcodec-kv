"""Archive serialisation (the storage/transport entropy coder).

1) save/load is bitwise exact for a CodecKVCache (packed 7-bit + profile) and
   for a FlashCodecCache rebuilt from a model config
2) the compressed file is smaller than the uncompressed payload, and the
   in-memory footprint is unchanged by the round trip
3) loading a multi-layer archive without a config is rejected (it would have to
   drop per-layer codec settings)
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flashcodec import (CodecKVCache, FlashCodecCache, HAS_HF_CACHE,
                        load_archive, save_archive)

torch.manual_seed(0)

B, H, S, D, R = 1, 4, 96, 32, 8
k_seq = torch.cumsum(torch.randn(B, H, S, D) * 0.1, dim=-2)
v_seq = torch.cumsum(torch.randn(B, H, S, D) * 0.1, dim=-2)
G = torch.rand(8, D) + 0.5
GV = torch.rand(8, D) + 0.5


def _tmp():
    return tempfile.mktemp(suffix=".fckv")


def test_roundtrip_codec_cache():
    c = CodecKVCache(mode="jpeg", block_size=R, keep_freqs=8, quant_bits=7,
                     channel_scale=G, channel_scale_v=GV)
    c.forward(k_seq, v_seq)
    path = _tmp()
    stats = save_archive(c, path)
    c2 = load_archive(path)
    assert torch.equal(c.key_cache[0], c2.key_cache[0])
    assert torch.equal(c.value_cache[0], c2.value_cache[0])
    assert c.storage_bytes() == c2.storage_bytes()
    assert torch.equal(c2.codec.channel_scale, G)
    assert torch.equal(c2.value_codec.channel_scale, GV)
    assert stats["stored_bytes"] < stats["raw_bytes"], stats
    print(f"   payload {stats['raw_bytes']/1e3:.1f}KB -> "
          f"{stats['stored_bytes']/1e3:.1f}KB ({stats['ratio']*100:.1f}%)")
    Path(path).unlink(missing_ok=True)


def test_roundtrip_8bit():
    c = CodecKVCache(mode="jpeg", block_size=R, keep_freqs=8, quant_bits=8)
    c.forward(k_seq, v_seq)
    path = _tmp()
    save_archive(c, path, level=1)
    c2 = load_archive(path)
    assert torch.equal(c.key_cache[0], c2.key_cache[0])
    Path(path).unlink(missing_ok=True)


def test_roundtrip_flash_cache():
    if not HAS_HF_CACHE:
        print("   (skipped: transformers 5.x not available)")
        return
    from transformers import LlamaConfig
    cfg = LlamaConfig(hidden_size=64, num_attention_heads=4,
                      num_key_value_heads=2, num_hidden_layers=2, head_dim=16)
    cache = FlashCodecCache.from_config(cfg, mode="jpeg", keep_freqs=8,
                                        quant_bits=7)
    k = torch.randn(1, 2, S, 16) * 0.1
    v = torch.randn(1, 2, S, 16) * 0.1
    cache.update(k, v, layer_idx=0)
    path = _tmp()
    save_archive(cache, path)
    c2 = load_archive(path, config=cfg, materialize=True)
    assert torch.equal(cache.layers[0].keys, c2.layers[0].keys)
    assert torch.equal(cache.layers[0].values, c2.layers[0].values)
    assert cache.storage_bytes() == c2.storage_bytes()
    Path(path).unlink(missing_ok=True)


def test_multilayer_without_config_is_rejected():
    if not HAS_HF_CACHE:
        print("   (skipped: transformers 5.x not available)")
        return
    from transformers import LlamaConfig
    cfg = LlamaConfig(hidden_size=64, num_attention_heads=4,
                      num_key_value_heads=2, num_hidden_layers=2, head_dim=16)
    cache = FlashCodecCache.from_config(cfg, mode="jpeg", keep_freqs=8,
                                        quant_bits=7)
    for layer in (0, 1):
        cache.update(torch.randn(1, 2, S, 16), torch.randn(1, 2, S, 16),
                     layer_idx=layer)
    path = _tmp()
    save_archive(cache, path, level=1)
    try:
        load_archive(path)
    except ValueError:
        pass
    else:
        raise AssertionError("multi-layer archive without config must be "
                             "rejected")
    finally:
        Path(path).unlink(missing_ok=True)


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
