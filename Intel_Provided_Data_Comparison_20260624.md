# Intel 提供数据对比分析

## 1. 对比口径说明

Intel 提供的截图数据是 `vLLM bench serve` 口径：

- 指标：`Output Token Throughput (tok/s)`、`Mean TTFT (ms)`、`Mean TPOT (ms)`
- 场景 1：`input_len=1024, output=512`
- 场景 2：`input_len=4000, output=1000`
- 配置：`2 x Arc B70, TP=2, FP8, GPU Memory Util=0.9`

本轮实测是 `test-model v2.6.0` 口径：

- 单路上下文测试：输出固定约 `1000 tokens`
- 并发测试：报告中的 TPS 是“单路平均 TPS”，不是系统总吞吐；系统总吞吐可粗略按 `平均 TPS x 并发数` 估算
- 配置：Intel 参数变量 + `gpu_memory_util=0.85` + `max_model_len=271360` + FP8 KV cache

因此两者不能完全等价，只能做近似趋势和量级对比。

## 2. Intel 参考数据

### 2.1 Intel: input=1024, output=512, TP=2

| Batch | Duration(s) | Output TPS | Mean TTFT(ms) | Mean TPOT(ms) |
|---:|---:|---:|---:|---:|
| 1 | 5.57 | 91.84 | 138.50 | 10.64 |
| 2 | 10.41 | 98.33 | 192.87 | 19.99 |
| 4 | 10.58 | 193.52 | 313.15 | 20.09 |
| 8 | 10.98 | 372.92 | 573.90 | 20.36 |
| 16 | 14.41 | 568.35 | 928.33 | 26.36 |
| 32 | 23.09 | 709.66 | 1577.37 | 42.00 |
| 40 | 27.75 | 737.99 | 1884.11 | 50.53 |

### 2.2 Intel: input=4000, output=1000, TP=2

| Batch | Duration(s) | Output TPS | Mean TTFT(ms) | Mean TPOT(ms) |
|---:|---:|---:|---:|---:|
| 1 | 11.07 | 90.30 | 348.35 | 10.74 |
| 2 | 20.81 | 96.12 | 609.37 | 20.22 |
| 4 | 21.43 | 186.67 | 957.20 | 20.48 |
| 8 | 22.79 | 351.02 | 1637.54 | 21.15 |
| 16 | 31.68 | 504.99 | 2821.23 | 28.83 |
| 32 | 52.77 | 606.41 | 5207.84 | 47.45 |

## 3. 单路近似对比

### 3.1 近似 input=1024 场景

Intel 为 `input=1024/output=512`；本轮 test-model 为 `context=1024/output=1000`，输出更长。

| 配置 | Input/Context | Output | TTFT(ms) | TPS / Output TPS | TPOT 估算(ms) | 对 Intel batch=1 TPS |
|---|---:|---:|---:|---:|---:|---:|
| Intel TP=2 | 1024 | 512 | 138.50 | 91.84 | 10.64 | 100% |
| 本轮 TP=2 | 1024 | 1000 | 149.00 | 67.20 | 14.88 | 73.2% |
| 本轮 PP=2 | 1024 | 1000 | 159.00 | 73.91 | 13.53 | 80.5% |

结论：近似 1K 输入下，本轮 TP=2 单路 decode TPS 约为 Intel 参考的 `73%`，PP=2 约为 `81%`。TTFT 与 Intel batch=1 接近，但 TPOT 明显高于 Intel 参考。

### 3.2 近似 input=4000 场景

Intel 为 `input=4000/output=1000`；本轮使用最接近的 `context=4096/output=1000`。

| 配置 | Input/Context | Output | TTFT(ms) | TPS / Output TPS | TPOT 估算(ms) | 对 Intel batch=1 TPS |
|---|---:|---:|---:|---:|---:|---:|
| Intel TP=2 | 4000 | 1000 | 348.35 | 90.30 | 10.74 | 100% |
| 本轮 TP=2 | 4096 | 1000 | 359.00 | 65.79 | 15.20 | 72.9% |
| 本轮 PP=2 | 4096 | 1000 | 279.00 | 73.21 | 13.66 | 81.1% |

结论：近似 4K 输入下，本轮 TP=2 单路 TPS 约为 Intel batch=1 的 `73%`；PP=2 约为 `81%`，且 TTFT 低于 Intel 参考值。

## 4. 并发吞吐近似对比

test-model 并发 TPS 是单路平均 TPS。下面用 `系统吞吐估算 = 平均 TPS x 并发数`，与 Intel `Output Token Throughput` 做粗略对比。

### 4.1 短上下文并发对比

Intel 参考是 `input=1024/output=512`；本轮最接近的并发项是 `context=512/output=1000`，输入更短、输出更长，仍非完全同口径。

| Batch/并发 | Intel TP=2 Output TPS | 本轮 TP=2 估算总 TPS | TP=2 / Intel | 本轮 PP=2 估算总 TPS | PP=2 / Intel |
|---:|---:|---:|---:|---:|---:|
| 1 | 91.84 | 67.66 | 73.7% | 74.27 | 80.9% |
| 2 | 98.33 | 88.62 | 90.1% | 116.00 | 118.0% |
| 4 | 193.52 | 153.88 | 79.5% | 177.08 | 91.5% |
| 8 | 372.92 | 255.36 | 68.5% | 253.20 | 67.9% |

结论：并发 1/4/8 下，本轮系统总吞吐估算低于 Intel；并发 2 的 PP=2 估算值高于 Intel batch=2，但由于上下文/输出长度和工具口径不同，不宜视为严格领先。

## 5. 256K 长上下文补充

Intel 截图没有 256K 输入数据，因此无法直接对比。以下为本轮 256K + 约 1K 输出结果：

| 模式 | 256K 单路 TPS | 256K 单路 TTFT | 256K 并发 8 单路 TPS | 256K 并发 8 TTFT |
|---|---:|---:|---:|---:|
| TP=2 | 59.70 | 27.269s | 9.14 | 128.113s |
| PP=2 | 40.37 | 15.029s | 9.46 | 67.854s |

结论：256K 单请求 decode TPS 是 TP=2 更高；但 PP=2 的 TTFT 更低，并发 8 时 PP=2 的延迟也明显更低。

## 6. 差异原因判断

1. **测试工具口径不同**：Intel 是 `vLLM bench serve` 系统级输出吞吐；test-model 并发 TPS 是单路平均 TPS。
2. **输出长度不同**：Intel 1K 输入场景输出 512 tokens，本轮 test-model 单路/并发多为 1000 tokens。
3. **显存利用率不同**：Intel 截图为 `GPU Memory Util=0.9`，本轮按 Intel 参数截图中的启动项并结合 256K 要求使用 `0.85`，KV cache 余量不同。
4. **max_model_len 不同**：Intel 截图参数为 `34816`，本轮为支持 256K+1K 改为 `271360`，长上下文服务配置会改变 KV cache 分配和调度行为。
5. **PP/TP 范围不同**：Intel 截图只提供 TP=2 数据；本轮额外验证 PP=2，PP=2 在短上下文和 TTFT 上表现更好。

## 7. 结论

1. 与 Intel batch=1 参考相比，本轮 TP=2 在 1K/4K 近似单路场景下约为 Intel 输出 TPS 的 `73%`。
2. 本轮 PP=2 在 1K/4K 近似单路场景下约为 Intel 输出 TPS 的 `80%-81%`，单路 decode TPS 高于本轮 TP=2。
3. 并发估算下，本轮 batch/并发 4 与 Intel 差距约 `8%-20%`；并发 8 差距约 `32%`。
4. 由于 Intel 未提供 256K 数据，本轮 256K 结果只能作为新增长上下文能力验证，不应与 Intel 1K/4K 峰值 TPS 直接对比。
5. 若需要严格对齐 Intel 图表，应使用 `vLLM bench serve` 重新跑 `input=1024/output=512/batch=1..40` 和 `input=4000/output=1000/batch=1..32`，而不是用 test-model 并发结果换算。

