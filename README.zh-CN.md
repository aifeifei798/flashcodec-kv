# flashcodec

> 把 Transformer 的时序 KV Cache 张量当作**高维多媒体图像信号**来压缩。

[![PyPI](https://img.shields.io/pypi/v/flashcodec-kv.svg)](https://pypi.org/project/flashcodec-kv/)
[![CI](https://github.com/aifeifei798/flashcodec-kv/actions/workflows/ci.yml/badge.svg)](https://github.com/aifeifei798/flashcodec-kv/actions/workflows/ci.yml)

[English](README.md) | [简体中文](README.zh-CN.md)

传统 LLM 研究把隐空间张量视为孤立标量因子；flashcodec 认为 `(B, H, S, D)`
布局里的 **seq 轴是一幅高维图像的行/列**——相邻 token 隐向量自相关、能量集中
在低频、差分后方差收缩（具体程度取决于模型/层/张量：真实 KV 实测统计与逐层
注意事项见「基准」一节）。于是数字多媒体领域的两大经典压缩理论可以
直接迁移：

| 模式 | 灵感 | 变换 | 性质 | 存储 |
|---|---|---|---|---|
| `mode="jpeg"` | JPEG | 分块正交 1D DCT-II，保留最低 K 个低频系数 | **有损频域** | `K/r`（4×~8× 显存节省） |
| `mode="png"` | PNG | 块首锚点（I-Token）+ 一阶差分 Δ，Top-M 裁剪 | **无损/近无损时域** | 稠密等大；残差稀疏性留给熵编码 |

`quant_bits=8`（仅 jpeg）把系数存成 int8 + 每个（块, 频率）一个 scale——在
`K/r` 之上再省真实字节：**k=2 时只占原始 KV 的 6.6%（15×），PSNR 46.9 dB**，
而 fp32 系数是 25% / 47.2 dB。

数学形式（块长 r，操作轴 dim=-2）：

- **JPEG**：$X_{\text{blk}} \in \mathbb{R}^{r \times D}$，$C = M_{\text{enc}} X_{\text{blk}}$，
  $M_{\text{enc}} \in \mathbb{R}^{K \times r}$，$M[j,n] = c_j \cos\frac{\pi}{r}(n+\tfrac12)j$，
  行正交归一（$K=r$ 时逐块无损）；重构 $\hat X = M_{\text{enc}}^\top C$。
- **PNG**：$\Delta_i = x_i - x_{i-1}$（$i \ge 1$），$x_0$ 为全精度锚点；
  重建 $\hat x_i = x_0 + \sum_{j=1}^{i} \Delta_j$（`cumsum`，纯线性、autograd 可微）。
  相邻隐向量自相关 ⇒ $\mathrm{Var}(\Delta)/\mathrm{Var}(X) \approx 0.03$（**30× 方差压缩**，
  实测），Top-M 裁剪后残差近零稀疏。

```
                    ┌────────────────────────────────────────────┐
  (B,H,S,D) ──append──▶  raw_buffer（未满 r 的尾部，原始高保真）  │
                    │        │ 满 r 个 token，批量归档（一次 GEMM │
                    │        ▼  原地切片写入）                    │
                    │  compressed_storage（倍增预分配）           │
                    │   jpeg: 系数 (B,H,capacity,K,D)             │
                    │   png:  deltas (B,H,capacity,r-1,D)         │
                    │        + anchors (B,H,capacity,D)           │
                    └───────────────┬────────────────────────────┘
                                    │ read()：有效区解码 + raw 尾
                                    ▼
                            重构 KV 视图 (B,H,S,D)
```

归档存放在**单块几何倍增缓冲区**里（std::vector 式：初始 16 块，装满翻倍）。
每满块只做一次原地切片写入（无反复重新分配）；`read()` 只解码**新增归档块**
（每个新块一次批量 GEMM / cumsum，落在追加式重构缓存里），不再每个 decode step
重解全部历史。若改成逐块碎片归档 + 逐归档循环解压，S=4096 的逐步解码
read 会因 kernel launch 开销慢约 170×（实测 31ms → 0.18ms）。

## 研究结论（2026-10）：与 llama.cpp KV 量化对比的负结果

我们试过把它从"存储原型"推进成 llama.cpp 的 KV 缓存替代品：沿 seq 轴做 DCT，
替换或叠加 llama.cpp 自带的行级 KV 量化（`-ctk/-ctv`）。四个模型（Qwen3-0.6B、
Hy-MT2-1.8B、Gemma4-E2B、纯密集 Qwen3-8B），同字节，指标是 **mean KL +
top-1 一致率**（实测发现微调模型上 ppl 是校准产物，而且它和 KL 对"K/V 谁更敏感"
给出相反答案）：

| 方案 | 8-bit 预算 | 4-bit 预算 |
|---|---|---|
| 截断（`k < r`） | — | 同等损伤下字节是 `q4_0` 的 2.7×；`k=2` 不可用 |
| 无损基 + 逐频 int8 | 损伤是 `q8_0` 的 8× | 13% 字节，不可用 |
| DCT 系数 + ggml 每 32 值 scale | 0.6B 上 KL 低 3×，1.8B/E2B/8B 持平 | 0.6B 上 KL 低 2.3×（仍然崩），其余 6–170% **更差** |
| codec 叠加在 `q8_0` 之上 | 少 0.8% 字节，多 63% 损伤 | — |

在真正重要的工况——`q8_0` K + `q4_0` V，占 fp16 KV 的 40.6%——Qwen3-8B 上这个
变换是**无效操作**（K 走 DCT 的 KL 0.0045 vs 原始 0.0047，top-1 97.2% vs 97.3%），
而"把 4 bit 花在 V 上而不是 K 上"的价值约为它的 10×。

**结论：没有在 llama.cpp 里实现，也不建议实现。** seq 轴变换只在行量化已经把
模型毁掉的区间才占优；凡量化本身可接受的工况，它都是持平或更差。留下来的价值
是机理层面的：变换把量化误差搬进位置间缓变子空间（L2 误差相同，ppl 损伤少 3–6×），
以及"KV 损伤取决于架构暴露面与 K/V 不对称，而不只取决于规模"。完整表格、
机理与复现见 [`docs/llama-cpp-kv-quantization.md`](docs/llama-cpp-kv-quantization.md)。

## 因果安全设计

只归档**完整块**；当前未满块永远留在 `raw_buffer`（原始精度）。增量解码时
任意时刻的未来扰动都不会改写已归档张量（测试逐位验证）。`mode="png",
keep_ratio=1.0` 的无损档位下，逐步解码注意力与全量前向注意力对齐 ≤1e-4。

## 安装

```bash
pip install flashcodec-kv            # 核心：torch>=2.0
pip install "flashcodec-kv[hf]"      # + transformers 5.x 后端
pip install "flashcodec-kv[demo]"    # + gradio/matplotlib（demo 用）
pip install -e ".[hf]"               # 或从源码安装
```

PyPI 分发包名：`flashcodec-kv`，导入名仍是 `flashcodec`（PyPI 上的
`flashcodec` 是无关占位包，别装错）。

## 用法

### 1. 独立使用 codec cache（不需要 transformers）

```python
import torch
from flashcodec import CodecKVCache

cache = CodecKVCache(mode="jpeg", block_size=8, keep_freqs=2, quant_bits=8)
k_hat, v_hat = cache.forward(k_seq, v_seq)   # 预填：(B,H,S,D) -> 重构视图
k1, v1 = cache.decode_step(k_new, v_new)     # 追加一个 token（别名 update）
print(cache.stats(0))        # tokens / archived / raw / storage_bytes / raw_bytes
print(cache.storage_bytes()) # 所有层的 archive + raw 尾
```

`forward` 会先重置该层再整体填充；`decode_step` 追加并在满块时归档。
要逐位无损档位就换 `mode="png", keep_ratio=1.0`。`crop(-n)` 删除尾部 token
（负数，遵循 HF 约定），`reorder_cache(beam_idx)` 做 beam search 的批维重排，
`reset()` 保留已分配容量。

### 2. transformers 5.x：显式传 cache

```python
from flashcodec import FlashCodecCache

cache = FlashCodecCache.from_config(
    model.config, mode="jpeg", block_size=8,
    keep_freqs=8,      # k = r：完整 DCT 基（变换本身无损）
    quant_bits=8)      # int8 系数 -> 约 bf16 KV 的 52%
out = model.generate(input_ids, max_new_tokens=64,
                     past_key_values=cache, use_cache=True)

print(cache.get_seq_length())        # prompt + new_tokens - 1
print(cache.storage_bytes())         # codec 层：archive + raw 尾
print(cache.codec_layer_indices())   # 例如 [3, 7, 11, 15, 19, 23]
```

`from_config` 只给 `full_attention` 层挂 codec，所以混合架构（Qwen3.5 式的
线性注意力 + 全注意力）里的线性层保持原生；纯 decoder-only 模型则每层都压缩。
`FlashCodecCache` 就是普通的 `transformers.cache_utils.Cache` 子类（5.x layer
API），`update` / `get_seq_length` / `crop` / `reorder_cache` 与原生 cache 同语义。

### 3. 一行热补丁

```python
from flashcodec import apply_flash_codec, remove_flash_codec

apply_flash_codec(model, mode="jpeg", block_size=8,
                  keep_freqs=8, quant_bits=8)
out = model.generate(input_ids, max_new_tokens=64)   # 自动注入 codec cache
remove_flash_codec(model)                            # 还原系统默认 cache
```

可用参数：`keep_freqs`（jpeg）、`keep_ratio`（png）、`quant_bits`（`0` 或
`2..8`，小于 8 时按位打包）、`channel_scale` / `channel_scale_v`（校准的通道
profile）、`per_layer={idx: {...}}`（逐层配置）、`materialization_free=True`
（见第 5 节）。

### 4. 档位怎么选

| 目标 | 配置 | 存储（bf16 KV） | 代价（Qwen3-0.6B 实测） |
|---|---|---|---|
| 输出逐位一致 | `mode="png", keep_ratio=1.0` | ~1.9× | 只花存储（fp32 差分） |
| **推荐（带校准）** | `keep_freqs=8, quant_bits=7` + profile | **~45%** | +0.025 ppl |
| 更省（带校准） | `keep_freqs=8, quant_bits=6` + profile | **~39%** | +0.249 ppl |
| 无校准安全档 | `keep_freqs=8, quant_bits=8` | ~52% | +0.52 ppl |
| 无校准 7-bit | `keep_freqs=8, quant_bits=7` | ~45% | +2.34 ppl |
| 截断（挂 codec 的层少时） | `keep_freqs=2` / `4` | 25–50% | 稠密模型直接崩 |
| 极限 | `keep_freqs=2, quant_bits=8` | ~13% | 稠密模型直接崩 |

经验规则：`keep_freqs = block_size`（完整 DCT 基）是前提；**降比特比截断频率划算**，
但必须先做通道 profile 校准，否则 7-bit 以下会崩。混合架构（只有少数层挂 codec）
对截断宽容得多，实测 k=4 甚至略好于基线。

#### 4.1 校准通道 profile（做 7-bit 的前提）

一行系数只有**一个** fp32 scale，却被 128 个通道共享；通道幅度差几个数量级，
量化级大部分被少数“响”通道吃掉。`calibrate_channel_scales` 用一次前向（1–2k
token）估一张静态 `(keep_freqs, head_dim)` 表，K/V 分开：

```python
from flashcodec import calibrate_channel_scales

ids = tokenizer(calib_text, return_tensors="pt").input_ids.to("cuda")
calib = calibrate_channel_scales(model, ids, keep_freqs=8)   # 一次前向
apply_flash_codec(model, mode="jpeg", keep_freqs=8, quant_bits=7,
                  per_layer=calib)
```

同一份 Qwen3-0.6B 实测（16×512 token；profile 用另外 2048 token 校准）：

| 档位 | ppl | Δ | 存储 |
|---|---|---|---|
| baseline | 31.1504 | — | 100% |
| k=8 b=8 | 31.6745 | +0.524 | 51.6% |
| k=8 b=8 + profile | 31.1210 | −0.029 | 51.6% |
| k=8 b=7 | 33.4916 | +2.341 | 45.3% |
| **k=8 b=7 + profile** | **31.1750** | **+0.025** | **45.3%** |
| k=8 b=6 | 47.2046 | +16.05 | 39.1% |
| k=8 b=6 + profile | 31.3993 | +0.249 | 39.1% |

混合架构 Qwen3.5-0.8B（只有 6/24 层挂 codec，baseline 23.5318）：b=7 + profile
+0.015、b=6 + profile −0.010。复现：`python benchmarks/bench_calibrated.py`
（默认用仓库自带 markdown 组窗口，校准切片与评测窗口不重叠）。

profile 是 codec 常量，解码只多一次 broadcast 乘：64K、k=8 时 0.371 → 0.414
ms/层；7-bit 的 gather 解包再翻一倍到 0.767 ms/层。`per_layer` 也可以只覆盖部分层。

#### 4.2 存档：熵编码只用于落盘 / 传输

热路径上 int8 系数直接给 attention 读，每步重新熵解码会亏；落盘时符号熵约
6.8/8 bit，压缩是净赚。`save_archive` 把逻辑 int8 流 + scales + tail 一起压缩
（装了 `zstandard` 就优先 zstd，否则 zlib）：

```python
from flashcodec import save_archive, load_archive

stats = save_archive(cache, "ctx.fckv")   # {"raw_bytes","stored_bytes","ratio"}
cache2 = load_archive("ctx.fckv", config=model.config, device="cuda")
```

28 层 Qwen3-0.6B、4001 token、k=8 b=7 + profile：内存 208.0 MB → 文件 181.6 MB
（逻辑载荷的 76.7%、bf16 KV 的 39.6%），逐位还原；save 5.0 s，load CPU 13.2 s /
`device="cuda"` 1.46 s（7-bit 重新打包在 GPU 上快约 100×）。

### 5. 免物化解码（不构建 KV 视图）

```python
apply_flash_codec(model, mode="jpeg", block_size=8, keep_freqs=8,
                  quant_bits=8, materialization_free=True)
out = model.generate(input_ids, max_new_tokens=64)

from flashcodec import codec_attention_stats
print(codec_attention_stats(model))   # 被包装的 attention 模块数（0 = 未启用）
```

decode step 直接用 Triton split-K kernel 在存储行上算 attention，并且释放
prefill 的重构视图，稳态显存只剩归档。该 patch 有守卫（单 token、batch=1、
eval、非滑窗、标准 `position_embeddings`），不满足会静默回退。

### 6. 诊断

| 调用 | 返回 |
|---|---|
| `cache.stats(layer)` | `tokens / archived / raw / storage_bytes / raw_bytes` |
| `cache.storage_bytes(layer=None)` | archive + raw 尾字节数（不含倍增预留容量与重构视图） |
| `cache.codec_layer_indices()` | 真正挂 codec 的层 |
| `codec_attention_stats(model)` | 被包装的 attention 模块数（`0` = 未启用） |
| `HAS_HF_CACHE` | transformers 后端是否可导入 |

### 7. Codec 级 API（不经过 cache）

```python
from flashcodec import JPEGCodec, PNGCodec

codec = JPEGCodec(block_size=8, keep_freqs=2, quant_bits=8)
coeffs = codec.compress(k_seq)              # (..., n_blocks, k, D)
q, scale = codec.quantize(coeffs)           # int8 + 每（块, 频率）scale
k_hat = codec.decompress(q, scales=scale)   # (..., S, D)，输入 dtype

png = PNGCodec(block_size=8, keep_ratio=1.0)
delta, anchor = png.filter(k_seq)
k_hat = png.reconstruct(delta, anchor)
```

张量契约：`(B, H, S, D)`，压缩轴 `dim=-2`。函数式 API（`dct_compress`、
`dct_decompress`、`quantize_coeffs`、`dequantize_coeffs`、`png_filter`、
`png_reconstruct`、`dct_basis`）同样导出。

### 8. 常见坑

- `pip install flashcodec-kv`，但 `import flashcodec` —— PyPI 名是分发包名，不是模块名。
- 从源码目录运行时，`import flashcodec` 必须解析到包目录而不是同名项目目录
  （`pip install -e .`，或进到项目目录里运行）。
- `crop` 收**负数**（`cache.crop(-4)` 删掉尾部 4 个 token）。
- beam search 各档都能用，但只有 `png keep_ratio=1.0` 与未打补丁模型逐位一致。
- `materialization_free=True` 只支持 decode 且 batch=1；批量/padding 推理会自动回退。
- `quant_bits` 只对 jpeg 有效（png 无损档必须保留 fp32 差分）。

## Transformers 集成（Qwen3.5-0.8B 实战验证）

混合架构（如 Qwen3.5：18 层线性注意力 + 6 层全注意力）自动处理——
`FlashCodecCache.from_config(model.config)` **只给 `full_attention` 层挂
codec 层**，其余层类型保留原生动态层（递归/卷积状态不受影响）。
`apply_flash_codec` 在拿到模型 config 时自动走 `from_config`。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from flashcodec import apply_flash_codec

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B",
                                             torch_dtype="bfloat16").cuda()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
ids = tok("杭州的西湖", return_tensors="pt").input_ids.cuda()

# 一行搞定——之后每次 generate() 都走编解码缓存
apply_flash_codec(model, mode="jpeg", block_size=8, keep_freqs=2)
out = model.generate(ids, max_new_tokens=48, do_sample=False)
```

Qwen3.5-0.8B 实测（bf16、CUDA、48 个贪心 token，`benchmarks/run_qwen_demo.py`）：

| 档位 | 存储 vs raw KV | 输出 | 结论 |
|---|---|---|---|
| `png keep=1.0` | ~1.8×（fp32 差分换精确）| **生成 token 与未打补丁模型逐位一致** | 100% 保真基线 |
| `jpeg k=8 int8` | **55.5%**（约 1.8×）| 文本流畅，**token 一致率 100%** | 安全有损档 |
| `jpeg k=2` | **32.3%**（约 3.1×）| 文本流畅，token 一致率 38.1% | 有损档 |
| `jpeg k=4` | 54.8%（约 1.8×）| 文本流畅，token 一致率 49.2% | 有损档 |
| `jpeg k=2 int8` | **21.1%**（约 4.7×）| 文本流畅，token 一致率 38.1% | 有损 + int8 |

混合层审计：`codec_layers=6/24`（仅全注意力层），零警告。复跑：
`python benchmarks/run_qwen_demo.py`（离线加载；支持 `FLASHCODEC_MODEL` /
`FLASHCODEC_NEW_TOKENS` 环境变量）。

dtype 契约（详见 `AGENTS.md`）：PNG 差分恒为 fp32（bf16 差分会破坏无损
档承诺的逐位精确性）；JPEG 系数与 `read()` 视图保持流输入 dtype
（SDPA 要求 q/k/v 同 dtype）。

## 免物化解码 attention

这里的编解码沿 seq 轴都是**线性映射**：一个块的 r 个 token 满足
`K_block = A @ S`，其中 `S` 是存储行（DCT 系数，或 anchor + 差分），`A` 是合成
矩阵。于是解码查询可以直接在 `S` 上算：

```
q · K[p] = (q · S^T) @ A[p]        out = (prob @ A) @ S
```

`codec_decode_attention(q, key_stream, value_stream)` 实现单 token 版本（jpeg
k<r、jpeg int8、png，含 GQA 分组）。`(S, D)` 的重构视图从不出现，结果与物化
路径差 ≤1e-6。

实测（`benchmarks/bench_attention.py`，B=1, H_kv=8, H_q=16, D=64, fp32）：

| seq | 模式 | 物化路径 | 系数域路径 |
|---|---|---|---|
| 16384 | jpeg k=2 int8 | 0.20ms / 130MB | 0.30ms / **32MB** |
| 65536 | jpeg k=2 int8 | 0.86ms / 520MB | **0.46ms / 128MB** |
| 65536 | png keep=1.0 | 0.87ms / 520MB | 1.03ms / 272MB |

适用范围：**仅解码**（prefill 仍走物化视图——系数形式会让 score 张量爆炸）。
png 每块存 r 行，数值上支持但省不了内存；收益来自 jpeg `k < r`。

**Triton split-K kernel**（`flashcodec/triton_attention.py`，flash-decoding 式：
按块分片写 partial max/sum/acc，再由 combine kernel 合并）去掉了中间张量和
launch 开销。单层、B=1、H_kv=8、H_q=16、D=128、`jpeg k=2 int8`
（`benchmarks/bench_attention.py --cuda`）：

| seq | 物化 | 系数域（torch） | 系数域（Triton） |
|---|---|---|---|
| 16384 | 0.20ms / 130MB | 0.29ms / 24MB | **0.09ms / 0MB** |
| 65536 | 0.86ms / 520MB | 0.41ms / 96MB | **0.16ms / 0MB** |

`apply_flash_codec(model, ..., materialization_free=True)` 已把它接进
transformers：包装标准 Llama/Qwen 风格 attention 模块的 forward（投影 + RoPE
不变，只替换 attention 输出），带单 token / batch=1 / eval / 非滑窗的守卫，
不满足条件自动回退原路径；prefill 的重构视图在第一个 decode step 释放。
`remove_flash_codec` 完整还原。

端到端（`benchmarks/bench_e2e.py`，Qwen3-0.6B，4096 token prompt + 32 token）：

| 档位 | tok/s | peak | steady |
|---|---|---|---|
| baseline（原始 KV） | 52.1 | 581MB | 0MB |
| codec k=2 int8（物化） | 44.2 | 1545MB | 1535MB |
| codec k=2 int8（免物化，Triton） | 40.3 | **642MB** | **122MB** |

**稳态显存 1/12.6**，解码只比物化路径慢约 9%（4096 token 下每 step：22.8ms vs
物化 21.7ms，原始 17.8ms）。剩余差距来自逐层的 split/combine 两次 launch，
persistent kernel 可以再吃掉。

## Demo（Qwen3.5-0.8B 左右对比 Chat）

![flashcodec Gradio 左右对比 demo](images/demo.png)

同一 prompt 两栏并排：左侧原生 KV Cache，右侧 flashcodec（五档：png 无损 /
jpeg k=8 int8 / k=4 / k=2 / k=2 int8），带「免物化解码」开关（系数域 +
Triton），每轮统计 KV 占用（归档 vs 物化视图）、峰值/稳态显存、每 token 延迟
与 token 一致率。

**Full comparison** 标签页会在同一 prompt 上跑 baseline + 所有档位，逐档给出：
KV 占用对比原始、归档/视图拆分、decode tok/s、峰值/稳态显存、续写部分的
teacher-forced NLL（困惑度式质量指标）与 token 一致率，并画存储/速度/NLL 三张柱状图。

```bash
pip install -e ".[demo]"   # gradio + matplotlib
python demo/qwen_gradio_chat.py    # Gradio 左右对比 + 节约柱状图
```

## 基准（实测，B=2 H=8 S=2048 D=64，随机游走自相关信号，fp32）

### 重构质量（`benchmarks/bench_reconstruct.py`）

| 配置 | PSNR (dB) | cosine | 存储比 |
|---|---|---|---|
| jpeg r=8 k=1（8×） | 42.95 | 0.99943 | 12.5% |
| jpeg r=8 k=2（4×） | 47.22 | 0.99985 | 25.0% |
| jpeg r=8 k=4（2×） | 51.66 | 1.00001 | 50.0% |
| jpeg r=8 k=2 int8 | 46.92 | 0.99983 | 6.6% |
| jpeg r=8 k=4 int8 | 50.89 | 0.99999 | 13.3% |
| png r=8 keep=0.5 | 50.11 | 0.99997 | 100%（稀疏留熵编码）|
| png r=8 keep=1.0 | ~196（fp 舍入级）| 1.00000 | 100% |

### 真实 KV 统计（真实模型，512 token）

`bench_reconstruct` 用的是合成随机游走；真实 cache 没那么光滑。在真实
post-RoPE KV 上实测（沿 seq 轴的 `Var(Δ)/Var(X)`，以及块长 8 时最低 k=2 个
DCT 系数保留的块能量占比）：

| 模型 / 层 | 张量 | Var(Δ)/Var(X) | k=2 保留能量 |
|---|---|---|---|
| Qwen3-0.6B L0 | K | 0.16 | 0.94 |
| Qwen3-0.6B L0 | V | 1.87 | 0.31 |
| Qwen3-0.6B L27 | V | 1.02 | 0.59 |
| Qwen3.5-0.8B L3 | K | 0.72 | 0.71 |
| Qwen3.5-0.8B L23 | V | 0.09 | 0.97 |

V 经常是**反**相关的（比值 > 1），所以「一阶差分让方差塌缩」是光滑合成信号
的性质，不是每一层真实 KV 的性质。实际结论：系数截断（`k < r`）必须按层决策
——稠密模型（28 层全压）会误差累积到崩，那里请用 `k = r` + `quant_bits=8`。

### 困惑度（`benchmarks/bench_ppl.py`；16 × 512 token，bf16，本地语料）

| 档位 | Qwen3.5-0.8B（压 6/24 层） | Qwen3-0.6B（压 28/28 层） | 存储 |
|---|---|---|---|
| baseline | 22.4803 | 16.7931 | 100% |
| png keep=1.0 | 22.4803 (+0.000) | 16.7931 (+0.000) | 188% |
| jpeg k=8（无损基） | 22.4877 (+0.007) | 16.8058 (+0.013) | 100% |
| **jpeg k=8 int8** | **22.4991 (+0.019)** | **17.1348 (+0.34)** | **~52%** |
| jpeg k=4 | 22.3010 (−0.18) | 470.85 (+454) | 50% |
| jpeg k=2 | 24.0566 (+1.58) | 6485.5 (+6469) | 25% |
| jpeg k=2 int8 | 24.0590 (+1.58) | 8118.5 (+8102) | 13% |

混合架构结果好是因为只有 6/24 层挂了 codec（18 层线性注意力保留原生状态）；
稠密模型那两列说明了为什么按层截断不能盲用。

### 逐层预算（`benchmarks/bench_adaptive.py`，Qwen3-0.6B，6 × 512 token）

校准方式：每次只把一层降到 k=4，测困惑度损伤；然后给损伤最小的那一半层
k=4、其余保持 k=r（75% 存储）：

| 方案 | ppl | Δppl | 存储 |
|---|---|---|---|
| baseline (k=r) | 18.9786 | — | 100% |
| 均匀 k=6 | 48.0531 | +29.07 | 75% |
| 自适应（14 层 k=4） | 40.8309 | +21.85 | 75% |
| 均匀 k=4 | 710.22 | +691.24 | 50% |
| 均匀 k=r int8 | 19.3627 | +0.38 | 50% |

同样存储下校准比均匀分配好 15%，但损伤是**超线性叠加**的（最便宜的 14 层
单独每个不超过 +0.5，合起来 +22），而 int8 在 50% 存储下好得多。所以把逐层
截断当最后手段；确实需要时用 `per_layer={idx: {"keep_freqs": ...}}`。

### 缓存占用与耗时（`benchmarks/bench_vram.py`，CPU）

| seq | 模式 | storage | 原始比 | fill | read |
|---|---|---|---|---|---|
| 4096 | jpeg k=2 | 4.0MB | 0.25 | 6.7ms | 0.0ms |
| 16384 | jpeg k=2 | 16.0MB | 0.25 | 34.3ms | 0.5ms |
| 65536 | jpeg k=2 | 64.0MB | 0.25 | 141.9ms | 3.1ms |
| 4096 | jpeg k=2 int8 | 1.1MB | 0.07 | 9.2ms | 0.0ms |
| 16384 | jpeg k=2 int8 | 4.3MB | 0.07 | 46.0ms | 0.3ms |
| 65536 | jpeg k=2 int8 | 17.0MB | 0.07 | 197.9ms | 3.1ms |
| 65536 | png keep=1.0 | 256.0MB | 1.00 | 243.3ms | 3.3ms |

CPU 计时逐次波动约 ±20%（取 3 次最小）；storage/ratio 两列是精确值。
`read` 为「物化视图并真实读取（`.sum()`）」的耗时，即消费端成本而非仅 kernel
启动。解码按归档块记忆化，`read` 只在**新增块**上做一次解码，不再每个 decode
step 重解全部历史（此前最后两行为 38.7ms / 71.9ms）。

PNG 模式不省稠密存储（无损基线的宿命）——它的价值在 30× 方差压缩 + Top-M
近零稀疏残差，是熵编码器的理想输入（本库不内置熵编码）。

## 与 llama.cpp KV 量化的同字节对比

llama.cpp 自带行级 KV 量化（`-ctk/-ctv`），所以公平的问题不是「codec 能不能压」，
而是「同样的字节预算下能不能赢过它」。同模型实测（Qwen3-0.6B、wikitext、
llama.cpp `184f65360`、CUDA + FA）：

| KV 配置 | ppl | Δ vs fp16 | 占 fp16 字节 |
|---|---|---|---|
| `-ctk f16 -ctv f16` | 19.6037 | — | 100% |
| `-ctk q8_0 -ctv q8_0` | 19.6138 | +0.05% | 53.1% |
| `-ctk q8_0 -ctv q4_0` | 19.6579 | +0.28% | 40.6% |
| `-ctk q4_0 -ctv q4_0` | 65.8321 | +236% | 28.1% |

对着这条线，flashcodec **当前归档格式**在同字节下没有优势：

- **截断（`k < r`）**：K 上 `jpeg k=4`（75% 的 fp16 K 字节）与全量 `q4_0`
  （28% 的 fp16 KV 字节）损伤相同 —— 同等损伤多用 2.7× 字节；密集模型上 `k=2` 不可用。
- **无损基 + `quant_bits=8`**（约 50% fp16 字节）：b32/k32 配置 +0.38% ppl，
  而 `q8_0` 是 +0.05% —— 同预算下损伤差 8×。
- **叠在 llama.cpp 量化之上也没有收益**：在 `q8_0` 之上 codec 只能再榨出 0.8%
  字节（每 128 个值一个 scale，而非每 32 个一个），代价是损伤 +63%；在 `q4_0`
  之上，int8 系数归档比它替换掉的 4-bit 载荷**更大**（39.8% vs 28.1%）。

**输的是系数量化粒度，不是变换本身。** 给系数用与 ggml 相同的每 32 值 absmax
（无损基、字节数完全相同），排序就反过来：**8-bit 下困惑度损伤少 5×、4-bit 下少 7×**，
而重构误差**一模一样** —— 变换只是把量化误差挪到位置间缓变（低频）子空间，
softmax 几乎看不见（误差整形，不是更好的 K）。该结果在 HF 侧测得，仍需在
llama.cpp 内确认；它也不能让本模型的 4-bit K 变得可用（ppl 41.6 vs 18.6）。

实测还显示 K 与 V 不对称：4-bit **K** 是唯一的悬崖（只压 K 到 `q4_0` → +237%），
4-bit V 几乎免费（+0.29%）。所以生产上便宜的配置是 `-ctk q8_0 -ctv q4_0`。
但这个悬崖是**小模型特性**：27B 混合架构（`qwen35`，17 个带 KV 的层）上
`q4_0` KV 在 8k 上下文只 +0.39%、32k 只 +0.20%，生产规模模型没有这个字节
预算问题需要解决。

**稠密中等模型是唯一"4-bit KV 有代价但不崩"的区间**，也是唯一 ppl 失效的区间。
Gemma4-12B（48 层，40 SWA + 8 全局 MQA）用 `--kl-divergence` 测得：8-bit KL
0.17、4-bit KL 0.30、top-1 一致率 89%（同一协议下 27B 混合只有 0.0035）。
它的 ppl 随语料在 2.2–355 之间漂移，量化还常常让 ppl *变好*，所以那一档
只能用 KL。

**随后的 HF 侧 KL 预判把这最后一个方向也关掉了**：同字节下，变换只在行量化
已经造成破坏时有效（Qwen3-0.6B：8-bit KL 低 3×，4-bit 低 2.3× 但仍然崩），
在 Hy-MT2-1.8B 和 Gemma4-E2B 上则是 8-bit 持平、4-bit 略差；而在**普通稠密
Qwen3-8B**（对 codec 最有利的架构）上，真正重要的工况完全没变化
（`q8_0` K + `q4_0` V、40.6% 字节：KL 0.0047 vs 加 DCT 的 0.0045，
top-1 97.3% vs 97.2%），双侧 4-bit 反而差 6%。不存在值得落地的工况，
因此**不跟进 llama.cpp 实现**。完整表格、机理与复现：
[`docs/llama-cpp-kv-quantization.md`](docs/llama-cpp-kv-quantization.md)。

## 为什么不做 V offload / 视频流解码（实测）

设想是：K 留在显存（高频、精确匹配），V 打包成 JPEG/AV1 丢到主机内存或硬盘，
用到时用 NVDEC 异步解压，从而打破 HBM 墙。三个实测把这条路堵死了
（`python benchmarks/bench_offload.py`；Qwen3-0.6B，RTX 5090 D，PCIe Gen4 x8，
pinned H2D 实测 10.7 GB/s）：

| 上下文 | decode 单步 | V（8-bit） | H2D 传输 | 传输/单步 | 与传输并行的单步 | NVDEC 每步解码 |
|---|---|---|---|---|---|---|
| 4K | 16.1 ms | 117 MB | 11.0 ms | 0.7× | 18.4 ms（1.1×） | 37.4 ms |
| 16K | 16.1 ms | 470 MB | 44.0 ms | 2.7× | 47.3 ms（2.9×） | 149.6 ms |
| 64K | 17.9 ms | 1879 MB | 176.1 ms | 9.8× | 177.4 ms（9.9×） | 598.3 ms |

1. **单步 decode 是 launch-bound，几乎不随上下文变长**（4K→64K 只有 16→18 ms），
   而 V 的传输量随 S 线性增长。传输/单步的比值**随上下文变长而恶化**，
   不存在"越长越划算"的拐点。
2. **瓶颈是 PCIe，不是 NVDEC**。每步用 NVDEC 解 V 是 10–33×；就算解码免费，
   光把 1.88 GB 像素搬进显存也要 176 ms。最后一列只是解压本身，还没算码流传输。
3. **"V 比 K 皮实"在 7-bit 下不成立**（`--kv-split`，同一套校准 profile）：
   只压 V（K 无损）−0.050 ppl、只压 K（V 无损）−0.078、两者都压 +0.003；
   6-bit 时只压 V +0.164、只压 K −0.022。K/V 不对称是 4-bit 的现象
   （见 llama.cpp 对比一节），在可用档位上两条流一样便宜。

所以：**≤4K 时 V offload 只贵 ~10%**（可接受，但省下的显存只有 235 MB）；
**≥8K 就开始超过单步（16K 2.9×、64K 9.9×）**。真正该做的是把 V 也压到 7-bit 留在显存
（45%，零 PCIe，+0.03 ppl），而不是搬到主机内存或视频流里。

唯一说得通的用法：把 V 真正落到**磁盘**做冷启动 / 跨机迁移（`save_archive`），
加载时后台解一次；不要把它放进每步的路径。

## 目录结构

```
flashcodec/
├── flashcodec/
│   ├── __init__.py        # apply_flash_codec / CodecKVCache / JPEGCodec / PNGCodec
│   ├── codecs/jpeg.py     # DCT-II 基底、投影/重构（纯 GEMM，无循环）
│   ├── codecs/png.py      # Anchor + 一阶差分、cumsum 还原、Top-M 阈值滤波
│   ├── cache.py           # CodecKVCache：raw 尾 + 倍增预分配归档，全量/增量双接口
│   ├── calibrate.py       # 逐层/逐流通道 profile（一次前向）
│   ├── storage.py         # save_archive / load_archive（zlib/zstd，落盘与传输）
│   ├── attention.py       # 免物化解码 attention（系数域计算）
│   └── patch.py           # 一行热补丁：Cache 层注入（HF Cache 鸭子类型）
├── tests/                 # jpeg / png / causal / attention / 校准位深 / 存档 六组测试
├── benchmarks/            # 显存/存储、PSNR/余弦、困惑度+校准、offload/NVDEC
├── docs/                  # 实测对比（vs llama.cpp KV 量化）
├── demo/                  # Gradio 左右对比 chat
├── images/                # README 截图
├── .github/workflows/     # CI：CPU 测试 + transformers 集成
├── CHANGELOG.md
├── pyproject.toml
└── README.md
```

## 测试

```bash
python tests/test_jpeg.py           # 正交归一、K=r 无损、k=2 余弦>0.99、K 单调
python tests/test_png.py            # 无损 maxdiff<1e-6、方差压缩比打印、可微
python tests/test_causal.py         # 未来扰动零泄露、增量 vs 全量 ≤1e-4、
                                    # 记忆化 read 与全量解码逐位一致、crop 精确且可重复、
                                    # beam search 批维重排
python tests/test_hf_integration.py # transformers 5.x 原生 generate 集成：
                                    # 显式缓存 / 热补丁 / PNG 无损逐 token 一致 /
                                    # 70 token 跨 8+ 块稳定性 / beam search 与 crop
                                    # 对未打补丁模型的精确性 / 逐层预算
python tests/test_attention.py      # 系数域解码 attention 与物化路径一致
                                    # （jpeg / int8 / png、GQA），且从不构建 KV 视图
python tests/test_calibrated_bits.py # 2..8 bit 位打包往返、校准 profile 降低误差、
                                    # 打包归档仍满足因果性、Triton == torch
python tests/test_storage.py        # 存档逐位往返（CodecKVCache + FlashCodecCache）、
                                    # 压缩后确实变小、多层归档必须带 config
```

## 稳定性（1.x）

1.x 期间公开 API 冻结：`apply_flash_codec` / `remove_flash_codec`、
`apply_codec_attention` / `remove_codec_attention`、`FlashCodecCache`、
`FlashCodecLayer`、`CodecKVCache`、`JPEGCodec`、`PNGCodec`、
`codec_decode_attention`、`make_codec`，以及 `quant_bits` / `per_layer` 选项。
新功能只做增量参数；既有签名默认值不变。下面的边界清单依然成立——这是一个
接口稳定的研究库，不等于生产就绪承诺。

## 边界与说明（诚实清单）

- 研究原型：`read()` 物化完整重构视图参与注意力，显存收益在**缓存存储层**
  （如上表）；与 flash-attention 类不物化内核结合才能量化端到端收益。该视图
  被记忆化并跨 decode step 复用，因此稳态显存 = 归档 + 一份原始 KV 大小的视图。
- 全量 `forward(seq)` 预填会把完整块直接压缩，块内位置的注意力会看到块级
  重构（块内近似）；增量 `decode_step` 路径无此问题（当前块恒为 raw）。
- 压缩张量为稠密布局；PNG 的 Top-M 稀疏性尚未接熵编码器（实测只做稀疏打包
  基本是白忙：fp32 差分本身占大头，bf16 输入下甚至可能比原始更大——得先做
  差分量化）。
- 系数截断（`k < r`）必须按层决策：稠密模型每层都压会误差累积（实测
  Qwen3-0.6B 困惑度 16.8 → 6486 @ k=2）。这种模型请用 `k = r` +
  `quant_bits=8`（约 52% 存储，+2% ppl）；激进压缩比留给混合架构（只有
  full_attention 层挂 codec 的情形）。
- **不是 llama.cpp KV 量化的更优替代 —— 至少不是当前归档格式**：同字节下
  `q8_0` 赢过两个可用档位（截断；以及无损基 + int8，+0.05% vs +0.38% ppl）；
  4-bit V 几乎免费，4-bit K 是唯一悬崖。但**变换本身**是另一回事：把系数交给
  ggml 式的每 32 值 absmax 后，8-bit / 4-bit 分别赢 5× / 7×，且重构误差不变
  —— 该结果在 HF 侧测得，尚未在 llama.cpp 内确认。详见
  [`docs/llama-cpp-kv-quantization.md`](docs/llama-cpp-kv-quantization.md)。
- 与 [Mini-Architecture](https://github.com/aifeifei798/Mini-Architecture)
  中 v4.2 原型（模型内 KV 复用层的 DCT/差分读取）同源：那里是"每位置视图"
  语义，这里是标准 cache 存储语义。

## License

MIT
