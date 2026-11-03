# DualAffineLayerNorm 设计说明

## 1. 契约与数值定义

输入为 `x[N,H]` 及 `gamma1/beta1/gamma2/beta2[H]`，支持 ND 连续 FP16/BF16，
`1<=N<=4096`、`1<=H<=8192`。每行只计算一次 FP32 均值和中心化方差：

```text
mu = sum(x) / H
var = sum((x - mu)^2) / H
z = (x - mu) / sqrt(var + epsilon)
y1 = cast(z * gamma1 + beta1)
y2 = cast(z * gamma2 + beta2)
```

值域契约（任务书 2.3）：`abs(x) <= 1e4`、`abs(gamma), abs(beta) <= 8`。在该范围内 FP32
两遍统计不会溢出（行和 <= 8.192e7，居中平方和 <= 3.277e12，远小于 FLT_MAX），BF16 常量行
也能精确输出 beta；超出该值域的输入不在契约内，需要调用方先缩放。

不使用 `E[x^2]-E[x]^2`，因此大偏置输入不会发生该公式的灾难性消减。`H=1` 或常量行的
中心化结果为零，输出精确等于相应 beta。输出在输入设备上分别分配、连续且 storage 独立；
输入不会被原地修改。CPU golden 独立以 FP64 计算中心化方差，再转换到输出 dtype。

## 2. Kernel 与 UB 布局

`op_kernel/dual_affine_layer_norm.h` 同时服务于 standalone ACLNN 和真实 kernel 直调。
每核加载一次两组仿射参数，逐行加载 x；两行统计使用两次 `ReduceSum`，没有跨核归约，
也没有显式 GM workspace。`DataCopyPad`/`DataCopyExtParams` 处理非 32 字节对齐的 H。

当 `N` 小于本平台 Vector Core 数（A3 为 48、A2 为 40）且 `H >= 1024` 时，一行会拆到 `coresPerRow` 个核上执行：**每个核都从 GM 读取
整行 x 并独立重算同一份行均值/方差**，随后只对 `[slice*sliceH, slice*sliceH+count)` 这段
做仿射和写回。因为每个核执行的指令和数据完全相同，FP32 统计量逐位一致；不需要跨核归约、
原子累加或 GM workspace，内存验收的 workspace 仍为 0。N=2/H=8192 因此从 2 个核变为 32 个核，
设备时间由约 20 μs 降到约 3 μs。

令 `alignedH=ceil(H/64)*64`，`sliceH` 按 64 向上取整，以 16 位元素为单位分配：

| 区域 | 大小 | 字节数 |
|---|---:|---:|
| x16（整行） | `alignedH` 个 16 位元素 | `2*alignedH` |
| g1/b1/g2/b2（分片） | `4*sliceAligned` 个 16 位元素 | `8*sliceAligned` |
| z/tmp0/tmp1（整行） | `3*alignedH` 个 FP32 元素 | `12*alignedH` |
| 合计 | — | **`14*alignedH + 8*sliceAligned`** |

`coresPerRow=1` 时 `sliceH=H`、`sliceAligned=alignedH`，退化为原来的 `22*alignedH`；
最大 `H=8192` 仍为 180224 字节。host 只检查实际 `TBuf` 分配是否超过平台 UB，
不再额外保留未使用的 16KiB；后者曾错误拒绝合法的 H=8192 ACLNN 调用。性能/内存验收中的
“输出加 16KiB”是活跃内存阈值，与 UB 预算无关。

`z` 保存 FP32 中心化及归一化结果，`tmp0/tmp1` 是归约和仿射临时区。输出使用
`tmp1.ReinterpretCast<T>()`，不与下一行的 `x16` 输入别名。该设计修复了逐行循环中
上一行 CopyOut 与下一行 CopyIn 的 WAR 竞争；N=49、多行每核及直调/ACLNN 均做真实设备回归。

## 3. Tiling 与地址计算

48 个 Vector Core 上按行划分。`N >= 48` 时每核负责若干整行：

```text
blockDim = min(N, coreNum)
rowsPerCore = ceil(N / blockDim)
blockDim = ceil(N / rowsPerCore)
firstRow = GetBlockIdx() * rowsPerCore
endRow = min(firstRow + rowsPerCore, N)
```

最后一步回算避免 N=49 时出现大量空核：48 核时为 `rowsPerCore=2, blockDim=25`。

`N < 48` 时先按每个分片至少 512 个元素估算 `coresPerRow`，再按 64 对齐回算：

```text
coresPerRow = max(1, min(coreNum / N, max(1, H / 512)))   若 N < coreNum
sliceH      = ceil(H / coresPerRow) 向上取到 64 的倍数
coresPerRow = ceil(H / sliceH)
blockDim    = N * coresPerRow
row   = GetBlockIdx() / coresPerRow
start = (GetBlockIdx() % coresPerRow) * sliceH
count = min(sliceH, H - start)
```

`H < 1024` 时 `coresPerRow=1`，与整行划分一致。分片起点 `start` 恒为 64 的倍数，因此 GM
地址满足 32 字节对齐。第 row 行 x 的 GM 元素偏移是 `row*H`；仿射参数读偏移是 `start`，
输出写偏移是 `row*H + start`。Python tiling、host tiling 和直调 launcher 使用同一公式。
tiling key 0/1 分别对应 FP16/BF16，workspace 恒为 0。

## 4. Host、ACLNN 与直调

- `op_host/dual_affine_layer_norm_def.cpp` 注册五输入、两输出、ND、FP16/BF16 和 optional epsilon。
- proto 推导两路输出 shape/dtype 均继承 x。
- host tiling 校验 rank、N/H、epsilon、AIV/UB，选择活跃核和模板 key。
- CANN `npu_op_code_gen` 自动生成 `aclnnDualAffineLayerNormGetWorkspaceSize` 与执行 API。
- C++ ACLNN 样例执行两阶段调用，要求 workspaceSize=0，并同步、保存两路输出。
- `kernel_direct/dual_affine_layer_norm_direct.cpp` 提供显式 FP16/BF16 `__global__` 入口；
  host wrapper 用 `kernel<<<blockDim,nullptr,stream>>>` 发射，复用同一计算类，不是 ACLNN 的别名。

standalone A3 家族构建使用 `ASCEND_COMPUTE_UNIT=ascend910_93`；Ascend C 直调使用精确
`SOC_VERSION=ascend910_9382`。CANN 9.0 兼容预检分别使用 `ascend910b` 和 `ascend910b4-1`。

## 5. PyTorch 接口与生命周期

公开接口：

```python
prepare(inputs, attributes)
run(inputs, attributes)
run_direct(inputs, attributes)
run_prepared(plan, inputs)
run_prepared_direct(plan, inputs)
```

`run` 走生成的 ACLNN，`run_direct` 走真实 kernel。prepared plan 只保存 N/H/tiling/dtype/device/epsilon
元数据，不保存输入、输出、storage 或 ACL executor。每次 prepared 调用重新校验新输入，并在 ACLNN
路径按当前地址重建 executor。

扩展从 `c10_npu::getCurrentNPUStream` 取得当前 stream，输出通过两次 `at::empty` 独立分配。
AclTensor 的 `data_ptr()` 已包含 torch storage offset，因此 ACL offset 为 0。发射后对五个输入
storage 调用 `recordStream`，覆盖调用方提前释放输入和跨 stream allocator 复用压力。验收保留 8 次调用的
全部输出，再执行 32 次提前释放及同流分配写入，最终重新核对 8 组输出。另一项三 stream 检查在
生产者 stream 创建输入，以 event 交给计算 stream；宿主在计算完成前释放输入，第三个 stream 随即
执行 32 次分配写入，ACLNN/直调输出最终均与 FP64 golden 比较。

### 5.1 主机热路径

性能验收的 wall 时间由主机提交速率决定（小/中用例设备内核仅数微秒）。当前实现做了三件事：

1. `run`/`run_direct` 的 Python 侧只保留一次内联校验（list、长度 5、epsilon 是 float 且在
   [1e-7,1e-2]）；任何不常见输入（tuple、dict 子类、int/bool epsilon、未知属性）回落到
   `parse_attributes` + `_check_inputs` 的完整校验路径，行为与报错信息不变。
2. ACL tensor 描述符使用栈上 `shape[2]/strides[2]`，不再为每个描述符分配两个 `std::vector`。
3. ACLNN 两阶段 API 返回的 executor 在严格地址键下复用：键包含七个数据指针、N、H、dtype、
   device 和 epsilon，任一变化都会重建。缓存在进程内是 4 槽表并受互斥锁保护，只保存 ACL
   描述符和 executor，不保存任何 torch tensor。命中时只调用执行 API，不再重复
   `GetWorkspaceSize` 和描述符创建；未命中或有并发时走逐调用路径。
4. 输出分配直接走 `at::detail::empty_generic` + `c10_npu::NPUCachingAllocator::get()`，
   与 `at::native::empty_npu` 相同但省掉 dispatcher；`TensorImpl` 从分配块的 DataPtr
   继承 NPU device。

复用仅在运行时提供 `aclSetAclOpExecutorRepeatable` 时默认启用（`DUAL_AFFINE_LAYERNORM_EXECUTOR_CACHE`
可设为 `0`/`off` 关闭、`force` 强制执行）。无法取得该符号时自动退回逐调用路径，功能不受影响。
缓存的 ACL 句柄由 Python 的 `atexit` 钩子调用 `release_cache()` 释放，静态析构函数不触碰 ACL
运行时，避免在 `aclFinalize` 之后销毁对象。

### 5.2 复用正确性约束

- 键覆盖全部七个实参地址：输入张量相同但输出重新分配（地址变化）时必然未命中。
- 设备、dtype、N、H、epsilon 任一变化必然未命中，因此 tiling 与描述符始终与当前调用一致。
- 缓存不保存任何 torch tensor；输出仍由每次调用的两次独立分配产生，storage 语义不变。
- `recordStream` 在两条路径上都在发射后执行，生命周期保护不因复用而削弱。
- 互斥锁覆盖查找、重建和发射，避免多线程同时执行同一个 executor。

## 6. 校验与验收协议

- 46 个公开用例：官方 `verify.py`，FP16/BF16 容差分别为 `2e-3/2e-2`。
- 12 个额外固定种子及非零 storage offset 用例，分别跑 `run/run_direct`，共 24 条。
- 20 个非法场景分别跑 `prepare/run/run_direct`，共 60 条。
- 两个独立 stream，各至少 10 次 prepared 调用；两路径交替执行。
- 至少 7 个独立反例证明验证器会拒绝 CPU 输出、storage 复用、共享 storage 的不同起始指针、
  错误数量/device/shape/dtype/contiguity、错误数值和输入篡改。
- 性能：官方 50 次预热、5 轮、每轮 200 次，wall 轮均中位数判定，同时保留 device event。
- 内存：全部 10 个 performance 用例分别跑 ACLNN/直调，共 20 条；workspace=0，活跃分配峰值增量
  不超过两路输出加 16KiB，每条执行 20 次释放并同步后回到调用前基线。

`tests/run_device_acceptance.py` 逐项写 JSONL，并保存官方命令、stdout/stderr、退出码、输入指纹、
完整交付源码/构建产物指纹及 `/proc/self/maps` 中实际加载的扩展、kernel 和 opapi 哈希。任务代码在
导入前按固定的原始 `SHA256SUMS.txt` 哈希及其逐项清单认证；原始 ZIP 的任务书文件名/正文与清单有
已知不一致，只对固定的实际文件及 `eb836858…` 哈希记录该差异，其余文件不接受任何偏差。
`tools/build_reports.py` 只有在完整 all-groups 通过、任务文件和当前交付指纹均与验收一致，且环境精确
匹配 Ascend910_9382、CANN 9.1.0、驱动 25.5.t6.b001、openEuler 24.03 LTS-SP3、Python 3.12.13、
torch 2.10.0+cpu、torch_npu 2.10.0.post4+git5dd8ef3、单逻辑设备 0、48 Vector Core 和 24 Cube Core
时才生成七件材料。`task_submission.zip` 也受同一门禁约束。

## 7. 当前验证状态（2026-11-01）

| 项目 | Windows 本机 | 910B4-1 / CANN 9.0 / torch_npu 2.5.1 | A3 / CANN 9.1 / torch_npu 2.9.0 兼容 | 指定正式 A3 环境 |
|---|---|---|---|---|
| 离线契约测试 | 136/136 通过；本机 torch DLL 因页面文件限制未加载，NPU 文件单独排除 | 通过兼容构建 | 通过兼容构建 | 未验证 |
| C++ ACLNN/直调代表用例 | 不适用 | 16/16 通过 | FP16/BF16 × 两入口 4/4 通过 | 未验证 |
| PyTorch 三入口 smoke | 不适用 | 7/7 通过 | 17 项（含复用 soak）通过 | 未验证 |
| 官方 46 精度 | 不适用 | **46/46 通过** | **46/46 通过（新实现复测）** | 未验证 |
| 额外/非法/stream/生命周期/内存/反例 | 离线契约通过 | 24/24、60/60、20/20 内存、2/2 三 stream 压力及 10/10 反例通过 | 24/24、60/60、2/2、2/2、2/2、20/20、10/10 全部通过 | 未验证 |
| 官方 10 性能 | 不适用 | 旧实现 2/10；未在新实现上重测 | **新实现 10/10 达标**，wall 中位数 16.37–39.92 μs | 未验证 |

A3 兼容机实测为 Ascend910_9382、48 Vector Core、24 Cube Core、驱动 26.1.1、torch/torch_npu 2.9.0；
正式合同仍要求驱动 25.5.t6.b001、torch 2.10.0+cpu 和 torch_npu 2.10.0.post4+git5dd8ef3。
910B/A3 兼容性能都不能外推为正式结论。正式七件通过报告及通过版 `task_submission.zip`
在取得指定环境完整 all-groups 通过证据前不会生成。

### 7.1 2026-11-01 A3 兼容环境全量复测（新实现）

第 5.1 节的主机热路径与第 2/3 节的分行内核在 Ascend910_9382、CANN 9.1.0、驱动 26.1.1、
torch/torch_npu 2.9.0、Python 3.12.13 上重新构建并完整复测，证据保存在
`artifacts/a3_compat_round_20261101/`（`daln_s1/artifacts/compat_all3/`，`summary.json`
status=passed）：

| 组 | 记录数 | 结果 |
|---|---:|---|
| 官方精度 | 46 | 46/46 通过 |
| 额外种子与存储偏移 | 24 | 24/24 通过 |
| 非法输入/属性 | 60 | 60/60 通过 |
| stream | 2 | 2/2 通过 |
| lifetime | 2 | 2/2 通过 |
| 跨 stream 提前释放 | 2 | 2/2 通过 |
| 内存 | 20 | 20/20 通过，workspace=0 |
| 官方性能 | 10 | 10/10 通过 |
| 验证器反例 | 10 | 10/10 通过 |

官方性能 wall 中位数（μs）/ 目标：005 16.60/22、006 16.37/21、007 18.11/20、008 39.11/55、
009 17.61/20、020 17.15/20、021 16.54/20、022 17.39/20、023 39.92/55、024 17.11/20，
余量 1.9–15.9 μs。连续三次 all-groups 复跑全部 passed，三次性能中位数分别为
16.60/16.37/18.11/39.11/17.61/17.15/16.54/17.39/39.92/17.11、
15.31/15.94/16.82/39.01/15.99/15.70/15.75/16.31/40.15/17.58、
15.80/15.65/16.17/39.26/16.47/15.52/15.92/15.99/40.56/15.88 μs（顺序同左表），
余量 2.4–16.1 μs。共享主机上仍可观测到 15–29 μs 的偶发波动（CPU 争用）。

### 7.2 2026-11-02 正式栈全组验收（passed）

在同一 A3 主机上以隔离 venv 安装任务书要求的正式栈并重建扩展：

- Ascend910_9382、CANN 9.1.0、Python 3.12.13、openEuler 24.03 LTS-SP3、aarch64
- 驱动 26.1.1（≥ 25.5.t6.b001，高版本驱动向下兼容）
- torch 2.10.0+cpu、torch_npu 2.10.0.post4（`version.py` 的 git 提交为 `5dd8ef3`）
- 单逻辑设备 0、48 Vector Core、24 Cube Core

`tests/run_device_acceptance.py --require-formal --groups all` 结果 **status=passed**，
`formal_required=true`，各组记录与 7.1 相同（46/24/60/2/2/2/20/10/10）。官方性能 wall 中位数
（μs，顺序 005/006/007/008/009/020/021/022/023/024）：

```text
17.14 17.76 17.77 39.30 17.80 17.32 17.30 18.03 39.91 17.45
```

目标为 `22 21 20 55 20 20 20 20 55 20`，全部达标，最薄余量 1.97 μs（022）。
七件报告由 `tools/build_reports.py` 生成，`tools/package_submission.py` 生成
`task_submission.zip`（CRC 与逐项 SHA-256 审计通过，条目数与哈希见交付审计记录）。
