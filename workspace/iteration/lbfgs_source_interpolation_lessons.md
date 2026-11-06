# Potential correction 经验：移动取样位置时，source 插值的梯度连续性很重要

实测日期：2026-11-04。体系：SDSS J0946+1006，Jackpot。
状态：实验经验暂存于本目录，供后续复用和统一整合；尚未改变库默认实现或形成 API 规范。

## 已验证的经验

**J0946 中，source 双线性插值在内部格线上的空间梯度跳变，是联合 L-BFGS 优化
停滞及终点数值检查不稳定的主要已验证原因。采用亮度和一阶导数均连续的 C1
插值后，受控对照中的收敛速度和重复稳定性显著改善。**

这里指插值表示的斜率不连续。双线性亮度在内部格线两侧连续；实验没有证明真实
source 的物理亮度本身不连续，也没有发现 JAX 在所选插值分支内算错导数。

后续复用这套梯度优化方案时，优先保证 source 前向表示及其一阶空间导数连续，
并同时检查外侧支撑边界。这是本次实测支持的选型经验，不是“所有双线性问题
都会失败”或“C1 能保证全局收敛”的定理。早先较简单的 Mock 使用双线性也能收敛。

## 为什么会影响 potential 的优化

Potential 系数 p 改变时，透镜偏折改变，图像像素在 source plane 的取样位置也移动：

```text
p 改变 → β(p) 改变 → 跨 source 格线 → 插值斜率跳变
       → 目标函数对 p 的梯度可能跳变 → L-BFGS 曲率估计、线搜索受干扰
```

固定宏模型下，可写成 `β(p) = β_macro − A p`，其中 A 是固定 potential 偏折算子。
图像对 p 的导数包含 `−∇S(β) · A`；PSF 卷积和子像素平均没有消除 source 格线的
斜率折点。L-BFGS 用相邻梯度变化估计曲率，在这种折点附近可能反复缩小步长、
耗尽迭代预算，或按目标值变化很小而停止，却未通过额外的光滑驻点检查。

JAX 自动微分会计算当前分支的正确导数，不会自动使前向模型光滑。只对返回的
梯度做平滑、而保留原双线性前向，会使梯度与目标函数不一致；本次没有采用该做法。

## 支撑结论的实测证据

### 外边界连续，还不足以修复内部折点

早期原模型在 source bbox 外直接截零，边缘亮度非零时会产生亮度跳变。外加一圈
固定零节点修复了该跳变，但双线性内部格线仍有斜率折点。真实数据的外边界/初值
对照仍有 12 次恢复未通过梯度检查，见[边界对照报告](fixed_boundary_results.md)。
该轮使用单步法的三项 λ，其中 λs≈2.28164e−4；不能与下述固定 M3 λs 的轮次混用数值。

### 内部格线复现与导数核对

固定 M3 source 超参数的旧重复中，两个终点的 potential 系数最大差仅约
2.12e−16，却使一条光线换到内部格线另一侧，GN 诊断从约 .000685 跳到 .0259。
分支内 AD 与独立 `Jwᵀ rw + Qq` 核对一致，详见
[旧终点分支核对](m3_source_fixed_results.md)。

在同一真实 source 和同一内部格线上，将跨线偏移从 ±1e−5 格缩到 ±1e−9 格：

| 插值 | ±1e−5 格的两侧梯度差范数 | ±1e−9 格 |
|---|---:|---:|
| 双线性零外环 | 107.2713 | 107.2713 |
| C1 cubic | .0055119 | 5.51e−7 |

这些是该 source、亮度单位和角坐标定义下的数值，不是通用阈值。C1 的两侧导数
差随偏移缩小而趋零；旧方法留下有限跳变。代码和原状态复算见
[格线探针](check_source_knots.py)与[C1 对照报告](cubic_source_results.md)。

### 同权重、同初值的真实恢复对照

`cubic_source_v1`：Float64，RTX 5080，JAX 0.7.1 / SciPy 1.17.1；2293 个数据像素，
100×100 source，1134 个 potential 系数、802 个 core，nsub=4。
两方法使用逐元素相同的冻结 source 初值、零 potential 和相同 hard-M3 白化参考。
宏模型、lens light、数据/noise/PSF/mask、自适应权重、先验和优化阈值不变。

最具区分力的一组固定强度为 λs=3.2741486396532473e−4、λ4=58605.388688316416、
λ0=53622.67084530563；source 亮度像素仍自由更新。

| 指标 | 双线性零外环 | C1 cubic |
|---|---:|---:|
| 三次通过数值验收 | 0/3 | 3/3 |
| L-BFGS 步数 | 均 3000 | 95–96 |
| 内层耗时中位数 | 31.15 s | .766 s |
| δκ 正峰 | .21414–.21416 | .21186 |

四组固定权重各三次：C1 合计 12/12 通过，双线性 7/12。C1 在上述组的三次整场
最大两两 κ 差约 5.71e−9。完整数字见[C1 对照报告](cubic_source_results.md)。

耗时为预热后的内层求解，不含一次性准备/JIT、白化因子、终点 evidence 和导出。
C1 该组加当次白化和终点核验约 1.56 s。替换插值也改变了前向函数和外侧支撑，
所以不能把耗时比解释为两个算法在完全相同目标上达到同一极小值的速度比。

## 本次 C1 表示的复用要点

实现为 [source_cubic.py](source_cubic.py) 的 `CubicSourceRenderer`：

- 张量积三次 Hermite，中心差分节点斜率，即 Catmull–Rom / Keys a=−.5。
  每条光线最多取 4×4 节点，系数仍表示原节点亮度；新增拟合参数为零。
- 原节点及 prior 不变，外侧系数固定零，支撑在外侧两格处以零亮度、零斜率结束。
  原双线性零环支撑为一格；这项边界定义变化属于新模型的一部分。
- 三次权重可为负、有局部过冲；本例沿用 signed source，没有另加亮度裁剪。
  “cubic”这个名字本身不保证外边界 C1，需单独核对所选延拓方式。
- 前向亮度、空间梯度、source 转置、potential Jacobian、AD/解析核对、终点 evidence
  使用同一插值定义。插值替换后，在当前终点重新计算 Jacobian。
- 白化参考只是优化坐标，不能代替新模型的前向或终点曲率。本例仍复用旧 hard-M3
  参考，未同步改变预条件器，从而保留受控比较。
- 复用其他插值基底时，重新核对系数是否仍是节点亮度、prior 的物理尺度和归一化；
  不把 B-spline 控制系数直接当成同一组节点亮度。

## 后续遇到停滞时的检查顺序

1. **先固定超参数排查内层。** 固定数据、网格、source 权重、初值、白化参考和精度，
   分开检查 source 外边界的亮度连续性与内部格线的一阶导数连续性。
2. **复算当前目标的导数。** 对比 AD 与解析梯度/伴随关系，并做跨线双侧步长探测；
   检查步长缩小后，亮度差和梯度差是否按预期缩小。单看图像形态或 success 不够。
3. **使用一致的前向表示修复。** 优先验证 C1 基函数及外侧延拓，避免只替换梯度。
   保留原先验定义；若改变了节点含义、网格或先验尺度，作为单独模型变更记录。
4. **恢复数值验收与重复检查。** 记录 native 退出原因、有限/SPD、当前 Jacobian 下的
   梯度诊断、目标/图像/完整 κ 场及 evidence。分开记录同初值技术重复和不同初值测试。
5. **再做外层 evidence 搜索。** 内层失败保留原始状态和 diagnostic evidence，明确正式
   分数的资格规则；否则大量失败点会限制外层实际可选择的区域。

本例数值规则为 native success、有限/SPD、`0.5 gᵀ H_GN⁻¹ g ≤ .001`；重复目标为
所有尝试通过、logE 范围≤.01、全 core 最大两两 κ 差≤1e−4。这些是本例的数值目标，
不是通用容差、真实目标误差上界、后验置信区间或检测显著性。

**未通过光滑梯度检查不等于恢复场错误。** 在非光滑折点的极小值处，左右坡度可能
没有平滑趋零；旧双线性结果中也存在 κ 图已经接近 C1 结果的情况。应分别描述
“跑满预算”“梯度检查未过”“场的重复差异”，不要统称为“物理恢复失败”。

## 数值修复与正则化选优要分开解释

后续两层优化均沿用 C1 和联合 L-BFGS，外层 DIRECT＋Powell，结果如下。
所有行都固定同一自适应 ρ 和 source_scale，source 亮度仍可更新。

| 模型/超参数条件 | λs | λ4 | λ0 | δκ 峰 | local-GN logE |
|---|---:|---:|---:|---:|---:|
| 固定单步 lens 权重、M3 λs | 3.27415e−4 | 5.86054e4 | 5.36227e4 | .21186 | 6298.1673 |
| 固定 M3 λs，优化 lens λ | 3.27415e−4 | 4.37901e5 | 8.22749e3 | .20057 | 6304.5840 |
| 联合优化 λs、λ4、λ0 | 2.53264e−4 | 5.38678e5 | 1.05727e4 | .17433 | 6307.4272 |

后两行各做原候选＋5次冷启动，共6次均通过；整场最大两两 κ 差分别约1.72e−8和
3.47e−8。三维搜索第二个 Powell 起点曾耗尽调用预算；后续独立近优点确认成功，
原退出记录未改。详见[二维搜索报告](cubic_evidence_results.md)与
[三维搜索报告](cubic_evidence_joint_results.md)。

这说明 C1 改善了本例的数值求解；峰值仍随 source–potential 的正则化权衡变化。
J0946 没有 injected κ 真值，不能按是否恢复 .2、峰是否更大或重复性好来证明真实
subhalo 质量。不同插值表示、不同权重的整个 κ 场也不必相同。

这里的 evidence 保留完整 source–potential 耦合，采用终点 local-GN 高斯近似；
不是 exact evidence 或完整 Hessian Laplace。λs 改变时需同步更新 prior、白化及
归一化中的 `Ns log λs`，不只改 penalty。独立小矩阵和真实数据核对均已覆盖这一点。
DIRECT/Powell 耗尽预算、访问过的最高分和 native 最终返回值也要分别保存。

## 复现入口和边界

从仓库根目录激活 `tinylens_gpu` 后，最小机制检查为：

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python workspace/iteration/check_source_knots.py --method bilinear --require-c1
# 上行预期断言失败：复现旧格线梯度跳变。
python workspace/iteration/check_source_knots.py --method cubic --require-c1
# 上行应通过。
```

真实数据拟合须使用全新的输出目录，程序拒绝覆盖已有非空目录；还需保留本地 M3 handoff：

- 同权重/初值比较：[run_cubic_source.py](run_cubic_source.py)。
- 固定 λs 的二维选优：[run_cubic_evidence.py](run_cubic_evidence.py)。
- 三项强度选优：[run_cubic_evidence_joint.py](run_cubic_evidence_joint.py)。
- 额外局部确认：[confirm_cubic_evidence_joint.py](confirm_cubic_evidence_joint.py)。
- 数学检查：[C1 测试](../../examples/tests/test_j0946_cubic_source.py)、
  [λs 与完整 evidence 测试](../../examples/tests/test_j0946_cubic_evidence_joint.py)。
  相关16项检查在原实验中通过；本文整理没有重跑拟合。

小型 Mock 增加 source 参数、减少观测数和降低信噪比的对照没有复现 J0946 的停滞，
参见 [source 网格](../pt_new_algorithm/source_resolution_results.md)、
[数据分辨率](../pt_new_algorithm/data_resolution_results.md)、
[噪声](../pt_new_algorithm/coarse_noise_results.md)。这些是各自 Mock 配置下的结果，
不能排除真实数据复杂度、预条件、初值、模型误差和退化之间的交互影响。

本经验支持把 C1 作为后续这套 L-BFGS potential-correction 实现的优先方案。
更换数据集、插值边界、source 节点或 prior 时，仍需重做对应的一致性与恢复验证。
