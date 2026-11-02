# DualAffineLayerNorm

双分支 LayerNorm Ascend C 算子：共享每行一次 FP32 均值和中心化方差，同时产生两组独立仿射输出。
仓库包含 standalone `.run` 包、自动生成的 ACLNN 两阶段 API、真实 kernel 直调、独立 C++ 样例、
PyTorch NPU 三入口及 prepared 接口，以及 fail-closed 验收与报告工具。

> **状态先行**：Atlas A3 Ascend910_9382/CANN 9.1 兼容环境的官方 46 个精度用例、扩展精度、
> 非法输入、stream、生命周期、内存和验证器反例均已通过，ACLNN/direct C++ FP16/BF16 样例 4/4 通过。
> 该环境使用驱动 26.1.1、torch/torch_npu 2.9.0，性能为 **2/10 达标、8/10 未达标**。
> 指定正式驱动 25.5.t6.b001 和 torch_npu 2.10 环境尚未验收，因此当前没有通过版七件报告或
> `task_submission.zip`。910B/CANN 9.0 的兼容结果同样保留。详见 [验证状态](#验证状态) 和
> [设计说明](docs/design.md)。

## 目录

```text
op_host/                  OpDef、shape/dtype 推导、tiling
op_kernel/                共享 Ascend C 计算内核
kernel_direct/            显式 FP16/BF16 kernel 直调入口
csrc/                     torch_npu 扩展和 prepared plan
examples/                 独立 C++ ACLNN 与直调样例
dual_affine_layernorm/    Python 公共接口
 tests/                   离线契约、真实 NPU 与完整验收脚本
tools/                    证据门禁、XLSX/日志、指纹与打包
solution.py               官方 verify.py/benchmark.py 加载入口
```

## 正式环境构建

目标：aarch64 openEuler 24.03 LTS-SP3、Atlas A3 Ascend910_9382、CANN 9.1.0、驱动
25.5.t6.b001、Python 3.12.13、torch 2.10.0+cpu、torch_npu 2.10.0.post4+git5dd8ef3。
验收还要求单逻辑设备 0、48 个 Vector Core 和 24 个 Cube Core。
先 source 正式 CANN 环境，不要覆盖其已有 `PYTHONPATH`：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /path/to/dual_affine_layernorm
bash build.sh --soc ascend910_93 -j 4
```

产物：

```text
build/custom_opp_*.run
build/packages/vendors/dual_affine_layernorm_custom/
```

只安装到用户目录（脚本拒绝系统 CANN 路径和覆盖已有 vendor 目录）：

```bash
bash install_private.sh \
  "$PWD/build/custom_opp_ubuntu_aarch64.run" \
  "$HOME/.local/dual_affine_layernorm_opp"
export ASCEND_CUSTOM_OPP_PATH="$HOME/.local/dual_affine_layernorm_opp/vendors/dual_affine_layernorm_custom${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
```

实际 `.run` 文件名以构建输出 `PACKAGE=...` 为准。也可直接把构建生成的
`build/packages/vendors/dual_affine_layernorm_custom` 用于扩展构建，不执行安装。

构建 PyTorch 扩展：

```bash
export DUAL_AFFINE_LAYERNORM_SOC=ascend910_9382
# 若使用上面的私有安装目录：
export DUAL_AFFINE_LAYERNORM_OPP_ROOT="$ASCEND_CUSTOM_OPP_PATH"
python setup.py build_ext --inplace
```

扩展、真实直调 kernel `.so` 和 `libcust_opapi.so` 一起放入 Python package，通过 `$ORIGIN` 加载。

## Python 使用

```python
import torch
import torch_npu
import dual_affine_layernorm as daln

# x: [N,H]；其余四个输入: [H]；全部为同设备、同 dtype 的 ND 连续 NPU tensor。
inputs = [x, gamma1, beta1, gamma2, beta2]
attributes = {"epsilon": 1e-5}

y1, y2 = daln.run(inputs, attributes)          # 自动生成的 ACLNN
y1d, y2d = daln.run_direct(inputs, attributes) # 真实 Ascend C kernel 直调

plan = daln.prepare(inputs, attributes)        # 只保存元数据
y1, y2 = daln.run_prepared(plan, inputs)
y1d, y2d = daln.run_prepared_direct(plan, inputs)
```

prepared plan 可按相同 N/H/dtype/device/epsilon 复用于新 tensor；不会保留旧输入或输出。
非连续、CPU、错误 rank/dtype/device/长度/N/H/epsilon/属性均会被拒绝。

## 独立 C++ 样例

先生成测试数据：

```bash
python tests/generate_npu_cases.py /tmp/daln_case --n 49 --h 33 \
  --dtype float16 --seed 20261029 --mode normal
cmake -S examples -B examples/build -DSOC_VERSION=ascend910_9382
cmake --build examples/build --parallel 4
```

ACLNN 样例参数为 `<case_dir> <device> <dtype> <N> <H> <epsilon>`；直调样例末尾还可指定 core 数。
运行后以独立 FP64 checker 校验：

```bash
export ASCEND_CUSTOM_OPP_PATH="$PWD/build/packages/vendors/dual_affine_layernorm_custom"
export LD_LIBRARY_PATH="$PWD/examples/build/lib:$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
examples/build/aclnn_dual_affine_layer_norm /tmp/daln_case 0 float16 49 33 1e-5
python tests/check_npu_outputs.py /tmp/daln_case
cp -a /tmp/daln_case /tmp/daln_case_direct
examples/build/direct_dual_affine_layer_norm /tmp/daln_case_direct 0 float16 49 33 1e-5 48
python tests/check_npu_outputs.py /tmp/daln_case_direct
```

`examples/build/lib` 必须加入加载路径，否则 direct 可执行文件找不到真实
`libdual_affine_layer_norm_direct.so`。两个样例使用独立输出目录，避免后一次运行覆盖前一次输出。

## 测试

开发依赖（只在离线测试与报告工具里需要，CANN 镜像默认不含）：

```bash
python -m pip install pytest openpyxl ml_dtypes
```

`ml_dtypes` 用于 bf16 用例的输入生成，`openpyxl` 用于生成七件 XLSX 报告。

本机离线测试（无 NPU 时不执行真实 NPU 文件）：

```bash
python -m pytest -q --ignore=tests/test_npu_binding.py
```

真实三入口 smoke：

```bash
python -m pytest tests/test_npu_binding.py -q
```

完整正式验收（先检查设备；脚本不含 `pkill`，不会清理其他任务）：

```bash
npu-smi info
bash tests/run_public_acceptance.sh /path/to/supplied_task_package "$PWD/artifacts/a3_run" 0
```

该命令要求正式环境并执行所有组。910B 等兼容预检可显式去掉正式门禁，但结果只标记兼容验证：

```bash
python tests/run_device_acceptance.py \
  --task-root /path/to/supplied_task_package \
  --output-dir artifacts/compat_run --device 0 --groups all
```

只有 `summary.json=passed` 且所有正式证据完整时才能生成报告和通过版提交包：

```bash
python tools/build_reports.py --run-dir artifacts/a3_run --output-dir task_submission
python tools/package_submission.py --run-dir artifacts/a3_run --output task_submission.zip
```

开发源码归档必须明确命名，不能冒充通过版：

```bash
python tools/package_submission.py --development --output dual_affine_layernorm_development.zip
```

## 验证状态

| 验证项 | 已取得的真实结果 |
|---|---|
| Windows 离线回归 | 129/129 通过；本机 torch 因页面文件不足无法加载，真实 NPU 文件排除 |
| 910B C++ 代表数值 | ACLNN/直调合计 16/16 通过 |
| 910B PyTorch 三入口 | 7/7 通过，含 BF16/FP16、storage offset、prepared 复用 |
| 910B 官方精度 | 46/46 通过 |
| 910B 扩展协议 | 24/24 额外、60/60 非法、双 stream、8+32 生命周期、20/20 内存、2/2 三 stream 压力、10/10 反例通过 |
| 910B 官方性能 | 2/10 达标，8/10 未达标；保留原始轮次，不作为正式性能结论 |
| A3 兼容构建 | standalone `.run`、torch 扩展、真实 direct kernel 和独立 C++ 样例构建成功 |
| A3 兼容 C++/PyTorch smoke | C++ ACLNN/direct × FP16/BF16 4/4；PyTorch 三入口 7/7 通过 |
| A3 兼容功能验收 | 官方 46/46、额外 24/24、非法 60/60、20/20 内存、2/2 三 stream 压力、10/10 反例及其余 stream/生命周期组通过 |
| A3 兼容官方性能 | **2/10 达标、8/10 未达标**；快路径复测 wall 中位数 25.86–40.94 μs，未放宽阈值 |
| 指定正式环境 | **尚未执行**；仍需驱动 25.5.t6.b001、torch 2.10.0+cpu、torch_npu 2.10.0.post4+git5dd8ef3 及完整 all-groups 通过 |

A3 兼容环境为 Ascend910_9382、CANN 9.1.0、驱动 26.1.1、torch/torch_npu 2.9.0、单逻辑设备 0。
原始兼容日志保存在 `artifacts/`（若已同步到本机）及远程任务目录。失败日志同样保留，不转换为通过。
A3 原始证据位于 `artifacts/a3_evidence_before_python_fix/`，当前快路径证据位于
`artifacts/a3_evidence_python_fastpath/`。本机 `build/` 与 Python 包内的 Linux `.so` 仍是从 q38 同步的
**910B/CANN 9.0/Python 3.10 兼容构建产物**，不是 A3/CANN 9.1 正式产物；A3 二进制保存在上述证据目录。

### 2026-11-02 正式栈验收通过（七件报告 + task_submission.zip 已生成）

在隔离 venv 中安装任务书要求的正式栈（torch 2.10.0+cpu、torch_npu 2.10.0.post4，
git 提交 `5dd8ef3`），按 torch 2.10 头文件重建扩展后执行
`tests/run_device_acceptance.py --require-formal --groups all`：**status=passed，
formal_required=true**，全部组记录通过。官方性能 wall 中位数（μs）：

```text
005 17.14  006 17.76  007 17.77  008 39.30  009 17.80
020 17.32  021 17.30  022 18.03  023 39.91  024 17.45
```

目标 `22 21 20 55 20 20 20 20 55 20`，10/10 达标。七件报告与
`task_submission.zip`（SHA-256 见交付审计记录）已生成并审计，副本在
`task_submission/` 与 `artifacts/`。

### 2026-11-01 A3 兼容环境复测（新实现 10/10）

上表 2/10 对应旧版扩展。当前源码的主机热路径、executor 复用、分配快路径与分行内核已在
Ascend910_9382 / CANN 9.1.0 / 驱动 26.1.1 / torch 2.9.0 上重新构建并完整复测：

| 组 | 记录 | 结果 |
|---|---:|---|
| 官方精度 | 46 | 46/46 |
| 额外种子/偏移 | 24 | 24/24 |
| 非法场景 | 60 | 60/60 |
| stream / lifetime / 跨 stream | 2 / 2 / 2 | 全部通过 |
| 内存 | 20 | 20/20，workspace=0 |
| 官方性能 | 10 | **10/10**，wall 中位数 16.37–39.92 μs |
| 验证器反例 | 10 | 10/10 |

证据：`artifacts/a3_compat_round_20261101/daln_s1/artifacts/compat_all3/summary.json`（status=passed）。
性能明细与目标：005 16.60/22、006 16.37/21、007 18.11/20、008 39.11/55、009 17.61/20、
020 17.15/20、021 16.54/20、022 17.39/20、023 39.92/55、024 17.11/20。

关键机制：

- `run`/`run_direct` 的 Python 侧只做一次内联校验，异常输入回落完整校验路径。
- ACLNN 两阶段 executor 在严格地址键（七个指针 + N/H/dtype/device/epsilon）下复用，
  未命中即重建；`aclSetAclOpExecutorRepeatable` 通过 `dlopen` 显式解析。
  关闭/强制：`DUAL_AFFINE_LAYERNORM_EXECUTOR_CACHE=0|force`。
- 输出使用 `at::detail::empty_generic` + NPU caching allocator，绕过 dispatcher。
- `N < 48` 且 `H >= 1024` 时整行拆到多个核：每核冗余重算同一行统计，无需 workspace，
  把 2x8192 从设备受限的约 20 μs 降到约 3 μs。

复测命令（设备上）：

```bash
python tools/perf_probe.py --task-root /path/to/supplied_task_package   --device 0 --entries run,run_direct --soak 16 --output artifacts/perf_probe.json
python -m pytest -q tests/test_npu_reuse_cache.py tests/test_npu_binding.py
bash tests/run_public_acceptance.sh /path/to/supplied_task_package "$PWD/artifacts/a3_run" 0
```
