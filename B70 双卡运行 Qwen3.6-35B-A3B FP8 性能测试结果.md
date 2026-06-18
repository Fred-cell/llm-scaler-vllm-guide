# B70 双卡运行 Qwen3.6-35B-A3B FP8 性能测试结果


## 1. 测试平台配置

| 项目 | 配置 |
|---|---|
| 测试服务器 OS | Ubuntu 24.04.4 LTS |
| 测试服务器 Kernel | `6.17.0-1009-intel`, boot cmdline includes `intel_iommu=off` |
| CPU | Intel Core Ultra 9 285K, 24 cores / 24 threads |
| GPU | 2 x Intel Graphics [0xe223], PCI 04:00.0 / 08:00.0 |
| GPU Driver Version | `E3C9808155FEF5AE5721170` |
| GPU Kernel Driver | `xe`, module srcversion `E3C9808155FEF5AE5721170` |
| GPU Firmware | GFX firmware status: normal; `bmg_guc_70.bin.zst`, `bmg_huc.bin.zst` installed |
| DDR | 3 x 24 GB DDR5, rated 5600 MT/s, configured 4400 MT/s |
| Memory | 70 GiB usable memory, 8 GiB swap |
| Level Zero / OpenCL | Level Zero 1.14.37435+12; OpenCL NEO 26.09.37435.12 |
| llm-scaler Image | `intel/llm-scaler-vllm:0.14.0-b8.3` |
| Model | `Qwen3.6-35B-A3B` |
| Quantization | FP8 online quantization |

驱动已按 Intel 文档 `2.3 Driver Offline Installation` 安装：

```text
multi-arc-bmg-offline-installer-26.18.8.2-combo
```

两张 B70 均可通过 `xpu-smi discovery` 和 `sycl-ls` 正常识别。

## 2. 服务启动配置

本轮测试分别验证了 PP=2 和 TP=2，均未显式设置 `--max-model-len`，服务返回：

```text
max_model_len = 262144
```

公共启动参数：

```bash
--quantization fp8
--dtype=float16
--gpu-memory-utilization 0.88
--max-num-batched-tokens 2048
--block-size 64
--disable-sliding-window
--language-model-only
--skip-mm-profiling
--enable-auto-tool-choice
--tool-call-parser qwen3_xml
--enforce-eager
```

PP=2 使用：

```bash
--pipeline-parallel-size 2
```

TP=2 使用：

```bash
--tensor-parallel-size 2
```

PC 平台额外增加以下 CCL/SYCL 宏配置：

```bash
CCL_ATL_TRANSPORT=ofi
CCL_SYCL_TMP_BUF_SIZE=268435456
CCL_ALLREDUCE=direct
CCL_CONFIGURATION=cpu_gpu_dpcpp
CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
CCL_SYCL_ALLTOALL_TMP_BUF=1
```

## 3. 性能测试结果

测试工具：`test-model v2.4.1`

测试内容：上下文长度性能、并发性能、流式输出、长输出等。

| 配置 | 平均 TPS | 128K TTFT | 128K TPS | 并发 1 TPS | 并发 2 TPS | 并发 4 TPS | 并发 8 TPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| PP=2 | 78.98 | 3.064s | 57.82 | 83.63 | 63.13 | 48.14 | 35.26 |
| TP=2 | 69.54 | 9.020s | 63.69 | 69.03 | 45.20 | 39.45 | 32.98 |

与更新前 `6.17.0-35-generic` 结果对比：

| 配置 | 更新前平均 TPS | 更新后平均 TPS | 更新前 128K TTFT | 更新后 128K TTFT | 更新前 128K TPS | 更新后 128K TPS |
|---|---:|---:|---:|---:|---:|---:|
| PP=2 | 79.05 | 78.98 | 3.070s | 3.064s | 57.44 | 57.82 |
| TP=2 | 68.95 | 69.54 | 10.692s | 9.020s | 63.74 | 63.69 |

当前观察：

- PP=2 整体平均 TPS 高于 TP=2。
- TP=2 在 128K decode TPS 上略高于 PP=2。
- 更新到 `6.17.0-1009-intel` 并增加 PC 平台宏后，PP=2 整体基本持平；TP=2 有小幅提升，128K TTFT 有改善。
- `--enforce-eager` 目前看起来仍是必要参数；去掉后 FP8 模型启动失败，日志中出现 `_xpu_C.fp8_gemm_w8a16` fake tensor / compile path 相关错误。

## 4. 预期性能确认

请 Intel 协助确认以下问题：

- 双 B70 运行 `Qwen3.6-35B-A3B` FP8 时，PP=2 和 TP=2 的预期 TPS / TTFT 基线分别是多少？
- 当前 PP=2 平均约 79 TPS、TP=2 平均约 69.5 TPS 是否符合预期？
- 128K 上下文下，PP=2 TTFT 明显低于 TP=2，但 TP=2 decode TPS 略高，这种表现是否符合当前 llm-scaler/vLLM 架构预期？
- PC 平台使用上述 CCL/SYCL 宏是否为推荐配置？是否还有其他推荐宏、BIOS 参数、kernel 参数或 CCL 参数？

## 5. 需要的技术支持

希望 Intel 协助确认或提供：

- B70 + `llm-scaler-vllm:0.14.0-b8.3` + Qwen3.6-35B-A3B FP8 的官方推荐启动参数。
- PP=2 / TP=2 在双 B70 上的推荐选择，以及不同上下文长度下的性能预期。
- `--enforce-eager` 是否必须保留；如果不保留，是否有推荐参数可避免 FP8 XPU 算子在 torch compile/fake tensor 路径下失败。
- 当前 `max_model_len=262144` 场景下，`gpu-memory-utilization`、`max-num-batched-tokens`、`block-size` 的推荐值。
- 是否建议升级到新版 `llm-scaler-vllm` 镜像，例如文档中提到的更新版本，以获得更好的 B70 性能。

如需要，我们可以继续提供完整测试环境复现方式或配合运行 Intel 指定 benchmark。

谢谢。

