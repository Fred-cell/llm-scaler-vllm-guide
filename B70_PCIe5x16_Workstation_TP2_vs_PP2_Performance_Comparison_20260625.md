# B70 PCIe 5.0 x16 工作站 TP=2 vs PP=2 性能对比报告

## 1. 测试对象

| 项目 | 配置 |
|---|---|
| 测试平台 | Intel B70 双卡工作站，PCIe 5.0 x16 |
| OS / Kernel | Ubuntu 24.04.4 LTS, kernel 6.17.0-1009-intel |
| CPU | Intel Xeon 636, 12C / 24T |
| GPU | 2 x Intel B70 / Intel Graphics [0xe223] |
| 测试工具 | test-model version 2.6.0 |
| 推理镜像 | intel/llm-scaler-vllm:0.14.0-b8.3.1 |
| vLLM 版本 | 0.14.1.dev0+gb17039bcc.d20260605 |
| 模型 | Qwen3.6-35B-A3B |
| 最大上下文 | max_model_len=271360，覆盖 256K 输入 + 1K 输出 |
| 对比口径 | TP=2 no-prefix vs PP=2 recheck |

说明：本报告优先采用 TP=2 no-prefix 报告与 PP=2 复测报告对比，因为两者均关闭 prefix caching，测试口径更接近。需要注意 TP=2 使用 `gpu_memory_util=0.9`，PP=2 使用 `gpu_memory_util=0.85` 且显式开启 `--kv-cache-dtype fp8`，因此不是完全严格的单变量对比。

## 2. 启动配置对比

### TP=2 no-prefix

```bash
--gpu-memory-util 0.9
--max-num-batched-tokens 8192
--max-model-len 271360
--tensor-parallel-size 2
--quantization fp8
--tool-call-parser qwen3_xml
```

关键日志：

| 项目 | TP=2 |
|---|---:|
| enable_prefix_caching | False |
| GPU KV cache | 249,856 tokens |
| 271,360 tokens/request 最大并发 | 3.65x |

### PP=2

```bash
--gpu-memory-util 0.85
--max-num-batched-tokens 8192
--max-model-len 271360
--pipeline-parallel-size 2
--quantization fp8
--kv-cache-dtype fp8
--tool-call-parser qwen3_xml
```

关键日志：

| 项目 | PP=2 |
|---|---:|
| tensor_parallel_size | 1 |
| pipeline_parallel_size | 2 |
| enable_prefix_caching | False |
| GPU KV cache | 319,488 tokens |
| 271,360 tokens/request 最大并发 | 4.59x |

## 3. 单流性能对比

| 上下文 | TP=2 TTFT | TP=2 TPS | PP=2 TTFT | PP=2 TPS | TPS 对比 |
|---|---:|---:|---:|---:|---:|
| 1K | 0.483s | 104.26 | 0.308s | 75.93 | TP +37.31% |
| 2K | 0.338s | 103.74 | 0.339s | 75.76 | TP +36.93% |
| 4K | 0.342s | 102.79 | 0.481s | 75.03 | TP +36.99% |
| 8K | 0.399s | 102.67 | 4.770s | 74.40 | TP +38.00% |
| 16K | 0.609s | 102.98 | 0.928s | 72.94 | TP +41.18% |
| 32K | 1.069s | 103.52 | 1.470s | 69.49 | TP +48.97% |
| 64K | 1.900s | 103.09 | 2.549s | 63.86 | TP +61.43% |
| 128K | 3.849s | 102.05 | 5.669s | 54.31 | TP +87.91% |
| 256K | 9.000s | 93.81 | 15.238s | 41.89 | TP +123.94% |

单流结论：

- TP=2 的 decode TPS 显著更高，1K-128K 基本稳定在 102-104 tokens/s，256K 仍保持 93.81 tokens/s。
- PP=2 随上下文增长下降更明显，1K 为 75.93 tokens/s，256K 降至 41.89 tokens/s。
- 256K 场景下，TP=2 TPS 约为 PP=2 的 2.24 倍；TTFT 从 PP=2 的 15.238s 降至 TP=2 的 9.000s。

## 4. 并发性能对比

### 1K 上下文

| 并发 | TP=2 TPS | TP=2 TTFT | PP=2 TPS | PP=2 TTFT | TPS 对比 |
|---:|---:|---:|---:|---:|---:|
| 1 | 104.28 | 0.218s | 75.79 | 0.297s | TP +37.59% |
| 2 | 54.41 | 0.309s | 54.09 | 0.317s | TP +0.59% |
| 4 | 55.72 | 0.362s | 44.56 | 0.498s | TP +25.04% |
| 8 | 52.59 | 0.506s | 31.50 | 0.546s | TP +66.95% |

### 32K 上下文

| 并发 | TP=2 TPS | TP=2 TTFT | PP=2 TPS | PP=2 TTFT | TPS 对比 |
|---:|---:|---:|---:|---:|---:|
| 1 | 102.50 | 1.051s | 69.95 | 1.391s | TP +46.53% |
| 2 | 56.08 | 1.602s | 63.10 | 1.963s | PP +12.52% |
| 4 | 53.34 | 2.586s | 43.16 | 2.969s | TP +23.59% |
| 8 | 43.72 | 3.962s | 32.51 | 4.386s | TP +34.48% |

### 128K 上下文

| 并发 | TP=2 TPS | TP=2 TTFT | PP=2 TPS | PP=2 TTFT | TPS 对比 |
|---:|---:|---:|---:|---:|---:|
| 1 | 102.20 | 4.188s | 54.55 | 5.679s | TP +87.35% |
| 2 | 51.85 | 6.388s | 35.20 | 8.168s | TP +47.30% |
| 4 | 41.27 | 11.518s | 29.38 | 13.236s | TP +40.47% |
| 8 | 26.60 | 20.967s | 17.90 | 23.580s | TP +48.60% |

### 256K 上下文

| 并发 | TP=2 TPS | TP=2 TTFT | PP=2 TPS | PP=2 TTFT | TPS 对比 |
|---:|---:|---:|---:|---:|---:|
| 1 | 93.43 | 9.197s | 42.56 | 15.257s | TP +119.53% |
| 2 | 46.47 | 15.016s | 32.85 | 22.672s | TP +41.46% |
| 4 | 27.83 | 24.301s | 17.61 | 37.764s | TP +58.04% |
| 8 | 16.70 | 49.524s | 9.99 | 70.242s | TP +67.17% |

并发结论：

- TP=2 在绝大多数并发场景下 TPS 与 TTFT 均优于 PP=2。
- PP=2 仅在 32K 并发 2 的 TPS 上出现局部领先，属于孤立结果，不改变整体趋势。
- 256K 并发 8 下，TP=2 TPS 为 16.70，PP=2 为 9.99，TP 约为 PP 的 1.67 倍；TTFT 方面 TP=2 比 PP=2 少约 20.7s。

## 5. 其他性能项

| 测试项 | TP=2 | PP=2 | 结论 |
|---|---:|---:|---|
| Streaming | 103.46 TPS / 0.309s TTFT | 80.71 TPS / 0.265s TTFT | TP TPS 更高，PP TTFT 略低 |
| Batch size 8 | 3.55 req/s | 2.67 req/s | TP +32.96% |
| Long output 1000 | 104.84 TPS / 0.199s TTFT | 77.79 TPS / 0.279s TTFT | TP TPS 更高 |

## 6. TP 与 PP 差异分析

TP，即 Tensor Parallel，是将单层模型计算切分到多张 GPU 上。每个 token 的每一层计算都可以同时使用两张 B70，因此 decode 阶段吞吐更强。代价是层内通信增加，例如 all-reduce 或 collective 通信。

PP，即 Pipeline Parallel，是将模型层按阶段切分到不同 GPU 上。请求需要依次经过多个 pipeline stage。对于单流、低并发或长上下文 prefill/decode 场景，pipeline bubble 较难完全隐藏，因此 TTFT 和 decode TPS 容易受到影响。

本轮数据与理论预期一致：

- B70 双卡通过 PCIe 通信，TP 确实存在跨卡通信开销，但当前 Qwen3.6-35B-A3B 测试中，TP 的双卡并行计算收益明显大于通信开销。
- PP=2 虽然提供更大的 KV cache 容量和更高理论最大并发，但 test-model 的并发 1/2/4/8 场景没有充分转化为实际吞吐优势。
- 长上下文 128K/256K 下，PP=2 的 pipeline 调度与阶段等待成本更明显，表现为 TPS 下降更快、TTFT 增长更高。

## 7. 结论与建议

1. 当前 B70 PCIe 5.0 x16 双卡工作站上，Qwen3.6-35B-A3B 性能优先建议使用 TP=2。
2. TP=2 在 1K-256K 单流和多数并发场景下均明显优于 PP=2，256K 单流 TPS 约为 PP=2 的 2.24 倍。
3. PP=2 的主要价值是更大的 KV cache 和理论并发容量，适合后续针对更高并发容量或混合并行策略单独调优。
4. 若目标是 256K+1K 长上下文性能，当前 TP=2 更符合性能预期；PP=2 暂不建议作为默认部署参数。
5. 后续如需严谨单变量验证，建议统一 `gpu_memory_util`、`kv-cache-dtype`、batch 参数后再进行 TP=2 与 PP=2 复测。

## 8. 参考报告

- `Intel_B70/b70_pcie5x16_workstation_v260_256k_noprefix_20260624/B70_PCIe5x16_Workstation_TestModel260_256K_NoPrefix_Performance_Report_20260624.md`
- `Intel_B70/b70_pcie5x16_workstation_v260_pp2_recheck_20260625/B70_PCIe5x16_Workstation_TestModel260_PP2_Recheck_Report_20260625.md`
