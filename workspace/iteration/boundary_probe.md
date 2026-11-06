# 最小 source 边界停滞复现

这是诊断用小例子，不是 J0946 拟合，也没有修改其 forward/优化结果。
使用真实 `lens_mapping_operator_bilinear_from`、Float64、JAX AD 与相同的
SciPy L-BFGS-B 设置。固定 source 亮度，仅优化一个回溯位移；初始 GN 白化
曲率严格为1，因此复现不需要高维病态条件数。

```bash
JAX_PLATFORMS=cpu python workspace/iteration/check_boundary_stall.py --boundary hard
JAX_PLATFORMS=cpu python workspace/iteration/check_boundary_stall.py --boundary zero-ring
```

第一条命令**预期 assertion 失败**，它是捕获此类停滞的检查，不是生产入口。
每个模式独立重置初值运行3次，当前三次结果分别完全相同：

| 边界模型 | L-BFGS 停止 | 步数 | 终点 GN proxy |
|---|---|---:|---:|
| 原硬截零 | 相对 objective 变化很小，梯度仍非零 | 11 | 0.05，未过1e-3 |
| 外侧固定零亮度环 | 梯度满足 gtol | 2 | 4.52e-31，通过 |

在原边界两侧，距离从1e-3缩到1e-7时：硬截断的亮度差始终为0.04；零环模型
的差从8e-4降到8e-8，按距离连续缩小。额外环的值固定为0，拟合参数仍只有一个。
新边界是一个明确的模型变体；得到更低 objective 本身不能证明其物理准确性。

本结果支持优先检验连续边界，但没有证明真实数据的全部失败都来自外沿，也没有
证明单靠零环能解决 J0946。双线性格点的梯度拐点、初值及预条件仍需独立检查。
后续应保持固定 λ/数据/宏模型做小范围对照，再恢复外层超参数搜索。
