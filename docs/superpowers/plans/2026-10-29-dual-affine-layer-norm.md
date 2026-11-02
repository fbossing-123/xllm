# DualAffineLayerNorm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a buildable CANN standalone DualAffineLayerNorm operator with ACLNN, direct-kernel and PyTorch NPU interfaces, plus evidence-gated verification and submission tools for Atlas A3.

**Architecture:** One row-per-AI-vector-core Ascend C kernel keeps a complete row and both affine parameter sets in UB, computes shared FP32 mean/variance with two reductions, then emits two independent outputs. Generated ACLNN code handles the standard two-stage API; a small C++ pybind adapter allocates fresh outputs and launches either ACLNN or the direct kernel on the current NPU stream. Python validates attributes, C++ validates tensor metadata, and real-device JSONL results gate all reports and submission archives.

**Tech Stack:** Ascend C, CANN OpDef/Optiling/ACLNN, CMake/C++17, pybind11, Python 3, NumPy, PyTorch/torch_npu, openpyxl.

**Spec:** `C:\Users\wzy85\Downloads\_extract_dual_branch_layernorm_20261029\题目1-双分支LayerNorm算子-独立包\题目1-双分支LayerNorm算子-独立包\双分支LayerNorm算子任务书.md`

## Global Constraints

- Operator name `DualAffineLayerNorm`; inputs `x,gamma1,beta1,gamma2,beta2`; outputs `y1,y2`; optional float `epsilon=1e-5`.
- FP16/BF16, ND contiguous `x[N,H]`; `1<=N<=4096`, `1<=H<=8192`, including unaligned H.
- Shared FP32 statistics per row; variance denominator H; no `E[x^2]-E[x]^2`; constant rows and H=1 produce beta exactly.
- No explicit global workspace. Outputs are independent contiguous same-device tensors and inputs are never modified.
- Formal target: Atlas A3 / Ascend910_9382, 48 vector cores, CANN 9.1.0, aarch64, Python 3.12.13, torch/torch_npu 2.10.
- Formal reports contain only real measurements. q38/910B/CANN 9.0 results are compatibility evidence, never A3 performance or acceptance claims.
- Build/install into user-selected directories; never overwrite system CANN.
- No commit, push or PR unless separately requested by the user.
- Public verifier package is read-only; generated source and tests live in this independent project.

## Review Focus

- Unaligned and tiny H: DataCopyPad plus zero-filled aligned tails must not affect statistics; test H=1,15,17,31,33,8191 in Tasks 2/5.
- Output storage aliasing: differing `data_ptr()` values can still share storage; test verifier counterexamples and independent retention in Task 5.
- Cross-stream lifetime: releasing inputs before completion must not permit allocator address reuse; test two streams and delayed input release in Task 5.
- Metadata errors: all `prepare/run/run_direct` paths must reject invalid dtype/device/rank/shape/attributes before launch; test 60 records plus seven verifier counterexamples in Task 5.
- Evidence integrity: missing/failed/mismatched results must not yield passing XLSX or ZIP; test report fail-closed behavior in Task 6.

---

### Task 1: Offline Contract and Test Harness

**Files:**
- Create: `dual_affine_layernorm/__init__.py`
- Create: `dual_affine_layernorm/attributes.py`
- Create: `dual_affine_layernorm/tiling.py`
- Create: `tests/test_attributes.py`
- Create: `tests/test_tiling.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `parse_attributes(attributes: object) -> float`.
- Produces: `make_tiling(n: int, h: int, core_num: int = 48, ub_size: int = 196608) -> dict` with `n,h,aligned_h,rows_per_core,block_dim,ub_bytes,workspace_size,tiling_key`.
- Later C++ behavior must match these validation/tiling contracts.

- [ ] **Step 1: Write failing attribute and tiling tests.** Cover defaults, only-epsilon dictionaries, bool/string/NaN/Inf/range rejection; H=1 and 8191 alignment; N/core balanced rows; maximum `22*aligned_h==180224`; zero workspace and tiling keys 0/1 reserved for FP16/BF16 dispatch.
- [ ] **Step 2: Run tests and verify failure.** Run `python -m pytest tests/test_attributes.py tests/test_tiling.py -q`; expect import errors before implementation.
- [ ] **Step 3: Implement the two pure-Python contracts.** Reject list attributes, unknown keys and non-finite values; use integer-only balanced row tiling and exact UB accounting.
- [ ] **Step 4: Run tests and verify pass.** Run the same pytest command; expect all green.

### Task 2: Ascend C Kernel, Host Registration and Standalone Package

**Files:**
- Create: `op_kernel/dual_affine_layer_norm.cpp`
- Create: `op_kernel/dual_affine_layer_norm.h`
- Create: `op_kernel/dual_affine_layer_norm_tiling_data.h`
- Create: `op_host/dual_affine_layer_norm_def.cpp`
- Create: `op_host/dual_affine_layer_norm_tiling.cpp`
- Create: `op_host/CMakeLists.txt`
- Create: `op_kernel/CMakeLists.txt`
- Create: `CMakeLists.txt`
- Create: `build.sh`
- Create: `install_private.sh`
- Test: `tests/test_kernel_contract.py`
- Test: `tests/test_npu_acl.cpp` (built/run by Task 3 sample infrastructure)

**Interfaces:**
- Produces: generated `aclnnDualAffineLayerNormGetWorkspaceSize(..., double epsilon, ..., uint64_t*, aclOpExecutor**)` and `aclnnDualAffineLayerNorm(void*, uint64_t, aclOpExecutor*, aclrtStream)`.
- Produces: `__global__ __aicore__ void dual_affine_layer_norm_half/bfloat16(GM_ADDR x,g1,b1,g2,b2,y1,y2,workspace,tiling)`.
- Tiling data fields exactly match `make_tiling`: signed 64-bit scalar fields for dimensions/row counts/UB and unsigned block dimension as required by generated tiling ABI.

- [ ] **Step 1: Write source-contract tests.** Assert the kernel uses two `ReduceSum` calls, `DataCopyPad`, no `E[x^2]` shortcut, no user workspace, FP32 centered variance, two independent output stores, and supported dtype dispatch. Assert OpDef names, optional epsilon and shape/dtype inference.
- [ ] **Step 2: Run tests and verify failure.** Run `python -m pytest tests/test_kernel_contract.py -q`; expect missing-file failures.
- [ ] **Step 3: Implement host registration and tiling.** Infer both outputs from x; require affine shape `[H]`; query AIV count and UB; choose `block_dim=min(N,core_num)` and balanced complete rows; align H to 64 elements; require `22*aligned_h+16384<=ub_size`; set zero explicit workspace and dtype-specific tiling key.
- [ ] **Step 4: Implement the kernel.** Use TBuf raw UB layout `x16,g1,b1,g2,b2,z32,tmp0,tmp1`; zero padded tails before reductions; read x once per row, cast FP32, reduce sum/mean, center and square, reduce variance, compute `rstd=1/sqrt(var+epsilon)`, retain z and share it across both affine branches. Use `RoundMode::CAST_RINT` for FP16 and the platform-supported BF16 cast mode; write exact byte counts with DataCopyPad.
- [ ] **Step 5: Implement standalone build and private install.** Follow CANN `npu_op_package/npu_op_code_gen/npu_op_library/npu_op_kernel_sources` patterns; fail on missing build products; install only under a caller-provided prefix and expose `ASCEND_CUSTOM_OPP_PATH` setup.
- [ ] **Step 6: Run offline tests and a real CANN compatibility build.** Run source tests locally; on q38 use only `/data1/wzy85_cannjudge_s10/dual_affine_layernorm`, source CANN environment, build `.run`, and record exact commands/log. Compilation or test failures remain failures until fixed.

### Task 3: Independent C++ ACLNN and Direct-Kernel Samples

**Files:**
- Create: `kernel_direct/dual_affine_layer_norm_direct.cpp`
- Create: `kernel_direct/dual_affine_layer_norm_direct.h`
- Create: `examples/aclnn_dual_affine_layer_norm.cpp`
- Create: `examples/direct_dual_affine_layer_norm.cpp`
- Create: `examples/sample_common.h`
- Create: `examples/CMakeLists.txt`
- Create: `tests/generate_npu_cases.py`
- Create: `tests/check_npu_outputs.py`

**Interfaces:**
- Produces: `LaunchDualAffineLayerNormDirect(dtype,n,h,aligned_h,rows_per_core,block_dim,epsilon,x,g1,b1,g2,b2,y1,y2,stream) -> void` used by both sample and pybind adapter.
- Both executables accept a case directory, read raw `x/gamma1/beta1/gamma2/beta2.bin`, write `y1/y2.bin`, and return nonzero on every ACL/ACLNN failure.

- [ ] **Step 1: Write deterministic binary fixtures and independent FP64 checker.** Fixtures retain dtype bit patterns; checker computes centered FP64 mean/variance, casts to output dtype, applies FP16 `2e-3` or BF16 `2e-2` tolerances and exact beta checks for constant/H=1 rows.
- [ ] **Step 2: Implement direct launch wrappers and standalone samples.** Direct mode constructs the same tiling values without calling ACLNN. Both samples allocate/copy device buffers, launch on an explicit stream, synchronize, copy outputs and clean up on all paths.
- [ ] **Step 3: Build and run on available CANN hardware.** Run representative FP16/BF16 aligned/unaligned/tiny/constant cases through both executables. Save logs and outputs; compare with the FP64 checker. State clearly that these are 910B compatibility results if run on q38.

### Task 4: PyTorch NPU Three-Entry Adapter

**Files:**
- Create: `csrc/dual_affine_layer_norm_torch.cpp`
- Create: `csrc/CMakeLists.txt`
- Create: `dual_affine_layernorm/extension.py`
- Create: `dual_affine_layernorm/__init__.py`
- Create: `setup.py`
- Create: `tests/test_python_surface.py`
- Create: `tests/test_npu_binding.py`

**Interfaces:**
- Produces: `prepare(inputs, attributes)`, `run(inputs, attributes)`, `run_direct(inputs, attributes)`, each returning `(y1,y2)`; prepare validates and returns an opaque reusable plan accepted by `run_prepared(plan,inputs)` and `run_prepared_direct(plan,inputs)`.
- C++ functions: `run_aclnn(vector<Tensor>,double)`, `run_direct(vector<Tensor>,double)`, `prepare(vector<Tensor>,double) -> opaque plan`, and prepared-run variants.

- [ ] **Step 1: Write failing Python-surface tests.** Use import/lib mocks to pin public signatures, default attributes, plan reuse and missing-extension error messages; do not fake NPU numerical success.
- [ ] **Step 2: Implement shared C++ validation and allocation.** Reject wrong arity/rank/dtype/device/contiguity/N/H/affine lengths. Allocate `at::empty` separately for y1/y2 on x's device; use current NPU stream and no explicit workspace. Prepared plans contain only validated immutable metadata, never retained input/output tensors.
- [ ] **Step 3: Implement both launch paths.** ACLNN uses the generated two-stage functions; direct mode launches the Ascend C entry. After launch, call `c10_npu::NPUCachingAllocator::recordStream` for every input storage on the current stream; check asynchronous errors through the normal stream synchronization path.
- [ ] **Step 4: Build and smoke-test the extension on NPU.** Build against installed torch/torch_npu headers without modifying system packages. Run `tests/test_npu_binding.py` for both paths and prepared plans; retain raw logs.

### Task 5: Full Device Acceptance and Adversarial Coverage

**Files:**
- Create: `tests/run_device_acceptance.py`
- Create: `tests/device_cases.py`
- Create: `tests/verifier_counterexamples.py`
- Create: `tests/run_public_acceptance.sh`
- Test: `tests/test_device_case_manifest.py`

**Interfaces:**
- Consumes all three public entries and the official `verify.py`, `benchmark.py`, `check_performance.py` from the supplied package via explicit paths.
- Produces JSONL files under `artifacts/<run_id>/raw/`: public accuracy, extra, invalid, stream, lifetime, memory, performance, environment and verifier-counterexample records with source/input fingerprints.

- [ ] **Step 1: Write manifest tests.** Pin 46 public cases, ten performance cases, 12 extra cases x two paths, 20 invalid scenarios x three entries, two streams x >=10 prepared calls, >=7 independent counterexamples, 20-release loops for both paths, and retained-eight-output/32-pressure-call lifetime checks.
- [ ] **Step 2: Implement device runner.** Generate fixed-seed inputs and FP64 goldens; verify shape/dtype/device/contiguity/storage independence, bit-identical inputs, numeric tolerances and exact beta rows. Exercise storage offsets and delayed input release. Record every case individually and fail the process if any required record is absent or failed.
- [ ] **Step 3: Implement performance and memory protocols.** Use official 50 warmup/5 rounds/200 calls wall timing plus device events for the ten target cases. Require zero explicit workspace, active allocation peak <= two outputs +16KiB and baseline restoration after 20 synchronized release cycles for each path.
- [ ] **Step 4: Implement adversarial verifier tests.** Include CPU output, input-storage output reuse, distinct starting pointers sharing one storage, wrong output count/device/shape/dtype/contiguity and input mutation. Confirm each invalid implementation is rejected rather than weakening the checker.
- [ ] **Step 5: Execute on formal A3.** Save environment, commands, stdout/stderr and exit codes. Any unavailable environment or failed case is reported as not verified/failed, never replaced by simulated data.

### Task 6: Evidence-Gated Reports, Design and Delivery

**Files:**
- Create: `tools/build_reports.py`
- Create: `tools/package_submission.py`
- Create: `tools/fingerprint.py`
- Create: `docs/design.md`
- Create: `README.md`
- Create: `tests/test_reports.py`
- Create: `task_submission/README.md` (status only, not fabricated acceptance evidence)

**Interfaces:**
- Produces the seven required files under `task_submission/` only from a complete passing run manifest.
- Produces `manifest.json` with SHA-256 of source/build artifacts, tool versions, test totals and compatibility/formal environment labels.

- [ ] **Step 1: Write failing report-gate tests.** Use synthetic fixtures explicitly labeled as test fixtures to prove missing, failed, wrong-environment and incomplete-count manifests are rejected; a complete synthetic manifest may produce files only in a temporary test directory and never a real passing submission.
- [ ] **Step 2: Implement XLSX/log/report generation.** Read JSONL verbatim into logs; create accuracy/performance/memory workbooks with per-case status and measured values. Refuse to overwrite a passing report with unverified content.
- [ ] **Step 3: Implement packaging and fingerprints.** Hash all delivered source, `.run`, libraries and generated reports; verify ZIP entries and hashes after writing. Packaging a passing `task_submission.zip` requires a complete formal A3 pass; otherwise emit only clearly named development source archives if requested.
- [ ] **Step 4: Write design and reproduction documentation.** Include formulas, exact UB `22*aligned_h` derivation, row/core and byte-offset math, two-stage/direct APIs, validation, stream lifetime, private installation, all test commands, performance/memory thresholds and a prominent verification-status matrix.
- [ ] **Step 5: Run all available offline tests and final artifact audit.** Run `python -m pytest -q`, verify no extracted task files were modified, and list separately: locally verified, 910B-compatible-verified, and A3-not-yet-verified items.

### Task 7: Independent Final Review

**Files:**
- Review all implementation and test files; modify only confirmed defects with regression tests.

- [ ] **Step 1: Request a fresh code/design review.** Focus on numerical edge cases, UB overlap/lifetimes, tiling ABI, ACL resource cleanup, stream ownership and report fail-closed behavior.
- [ ] **Step 2: Reproduce every finding before changing code.** Add a failing test where practical; otherwise document the exact compiler/device reproduction.
- [ ] **Step 3: Re-run affected tests and delivery audit.** Report remaining target-device work explicitly; do not claim formal acceptance without the real A3 evidence.
