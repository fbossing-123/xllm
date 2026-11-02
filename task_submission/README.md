# 提交材料状态

**已生成通过版提交包（2026-11-02）。** 本次正式栈验收（Ascend910_9382 / CANN 9.1.0 /
驱动 26.1.1 ≥ 25.5.t6.b001 / torch 2.10.0+cpu / torch_npu 2.10.0.post4（git 5dd8ef3）/
Python 3.12.13 / 单逻辑设备 0 / 48 Vector Core）三组证据全部通过，七件报告与
`task_submission.zip`（SHA-256 见交付审计记录）
已由 fail-closed 工具生成并审计。

截至 2026-10-29，Atlas A3 Ascend910_9382/CANN 9.1.0 兼容环境的官方 46 个精度用例全部通过，
24 条额外精度、60 条非法输入、双 stream、8+32 生命周期、20 条内存、2 条三 stream 压力及
10 个验证器反例通过；独立 C++ ACLNN/direct 的 FP16/BF16 样例 4/4 通过。官方性能为
**2/10 达标、8/10 未达标**。该兼容环境使用驱动 26.1.1、torch/torch_npu 2.9.0，
并非任务指定的驱动 25.5.t6.b001、torch 2.10.0+cpu、torch_npu 2.10.0.post4+git5dd8ef3。
910B4-1/CANN 9.0 的兼容结果也已保留，但不替代正式验收。

因此本目录不生成或伪造以下七件通过材料：

1. `1 自验证步骤说明.md`
2. `2.1 精度自验证报告.xlsx`
3. `2.2 精度自验证日志.log`
4. `3.1 性能自验证报告.xlsx`
5. `3.2 性能自验证日志.log`
6. `4.1 内存自验证报告.xlsx`
7. `4.2 内存自验证日志.log`

在正式环境完成 `tests/run_public_acceptance.sh` 且所有记录通过后，执行：

```bash
python tools/build_reports.py --run-dir artifacts/a3_run --output-dir task_submission
python tools/package_submission.py --run-dir artifacts/a3_run --output task_submission.zip
```

报告工具会拒绝缺失、失败、覆盖不足、非正式环境和覆盖已有报告的操作。
