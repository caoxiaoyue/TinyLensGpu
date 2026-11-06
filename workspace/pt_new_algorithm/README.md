# Potential correction：JAX 自动微分优化器实验

**三超参数 evidence 搜索：** 新增 [DIRECT 外层＋L-BFGS 内层](run_lbfgs_direct.py)，64 位、固定观测 source-only 初值，联合选择 lambda_s/lambda4/lambda0。原 200×200 Mock 的两例各 305 次 DIRECT 调用、耗时约 8.95/3.35 分钟，均以预算结束。有扰动峰值从 0.304 升至 0.477，但整体 κ RMSE 变差；control κ RMS 约降 7 倍。使用既有 local-GN evidence 近似、完整归一化与终点曲率，未接入核心 API。详见 [实验报告、图和复现](lbfgs_direct_results.md)。

**混合精度补测：** H0 构造/Cholesky 保留 64 位，物理 forward/AD 梯度使用 32 位；两种白化应用精度各做 22 次恢复，本轮均未再出现分解失败。有扰动 warm 约 0.590/0.609 秒，对比本轮纯 64 位 0.970 秒；对照全部达原验收，有扰动场差达标、目标函数精度仍未达原容差。详见 [混合精度报告](lbfgs_mixed_precision_results.md) 与 [独立脚本](benchmark_lbfgs_mixed_precision.py)。

**32/64 位补测：** 同一标准 200×200、seed=17、有扰动 Mock 的完整 warm 恢复，float64 中位数 0.988 秒，JAX float32 为 0.223 秒（观察值约快 4.42 倍）。32 位恢复形态与峰值接近，但未达原 64 位目标精度，额外 control 重复中出现一次初始白化分解失败；当前仍建议 64 位默认。计时含白化、优化、最终诊断/场产品，不含 Mock 生成。详见 [精度与速度报告](lbfgs_precision_results.md) 和 [独立脚本](benchmark_lbfgs_precision.py)。

**最终选型与稳定性：** [10个独立噪声种子的端到端验证](lbfgs_stability_results.md)已完成。观测源初值下，10次有扰动恢复和10次匹配control全部SciPy成功、参考一致性/物理检查通过，旧梯度门也20/20通过；无重跑或更换种子。后续采用 **JAX自动微分＋一次初始曲率白化＋L-BFGS-B** 作为本研究的默认方案，本轮算法实验收尾。范围仍为当前固定宏模型/网格/先验，系统接口未在本轮改动。

**GN性能补测：** 已按导数动作、收敛检查、传输及白化拆解，并做三轮单项/组合对照。可选fast组合使原200×200有扰动warm中位由8.96秒降至2.71秒、对照由2.24秒降至1.16秒，最终验收门保持原值。详见 [性能报告与复现命令](gauss_newton_performance_results.md)；默认仍保留初版设置。

新增 [SciPy高斯–牛顿原型](run_gauss_newton.py) 与 [实验报告](gauss_newton_results.md)：使用TRF/LSMR和JAX的Jacobian动作，原200×200测试的6次truth初值运行、2次观测源初值运行均通过原梯度验收。truth初值warm约8.79–9.07秒（有扰动）、2.22–2.24秒（对照）；比原dense LM快，仍慢于L-BFGS成功样本。该方法带信赖域保障，与论文的直接完整步更新分别说明。

另有独立的 [lmfit Levenberg–Marquardt 文件](run_lmfit.py) 与 [实验记录](lmfit_results.md)，保持原目标与JAX导数。原200×200对照通过原梯度验收、用时421.5秒；有扰动案例在600秒软预算后仍未通过、实际用时680.5秒。当前lmfit的CPU求解明显更慢，暂不建议替换现有方案；原生停止与梯度回调停止分别记录。

2026-11-03。独立、可丢弃的数值实验；暂不接入库的建模或推断接口。

**L-BFGS 结论：原始 200×200 完整 Mock 已通过，且仅由观测反演初始源的版本也通过。** 扰动、源和图像均与原 LM 基本一致，详见 [完整 Mock 实验报告](scientific_results.md)。方法包含一次初始自动微分曲率白化；仍不宣称任意问题都更稳定。

完整实验入口为 [run_scientific.py](run_scientific.py)，默认测试两种初值和 subhalo/control 两幅图；原始记录保留在 scientific_full_truth/ 和 scientific_full_source_only_v2/。以下是最初的小网格原型与历史结果。

**速度补测：** 与原 dense LM 相比，成功 warm 样本约快 20–30 倍，包含白化准备与终点检查。但有扰动重复中有未通过收敛门限的记录，简单收紧 ftol 尚未解决，因此不宣称稳定的整体加速比。详见 [完整计时结果](timing_results.md)。

## 首轮小网格运行

在仓库根目录使用现有环境，无需安装额外依赖：

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python workspace/pt_new_algorithm/run_experiment.py
```

只重现小型非线性案例：

```bash
python workspace/pt_new_algorithm/run_experiment.py \
  --cases nodal --scalings initial_whitened \
  --output workspace/pt_new_algorithm/results_nodal_recheck
```

`--scalings` 还支持 `raw`、`diagonal`、`whitened`、`initial_diagonal`。
自定义实验宜指定新的 `--output`，便于保留已有记录。GPU 浮点归约和分支选择可能让末位数值及迭代次数有小幅变化。

入口：[run_experiment.py](run_experiment.py)；固定问题构造：[problems.py](problems.py)；数学与官方接口核对：[algorithm_review.md](algorithm_review.md)。默认完整结果保存在 [results_final/summary.json](results_final/summary.json) 和 [summary.csv](results_final/summary.csv)。

## 实际优化什么

联合变量是源节点亮度和势修正节点，\(x=(s,p)\)。目标保持现有迭代法的定义：

\[
F(x)=\frac12\|[d-f(s,p)]/\sigma\|^2
+\frac12\lambda_s s^T R_s s
+\frac12\lambda_p\|D_4p\|^2
+\frac12\lambda_a\|p\|^2.
\]

每次函数评估都重新计算势修正后的射线、双线性源插值、子像素积分和 PSF；没有冻结首次线性化。宏模型、网格、mask、PSF、初值和三个正则化强度在同一案例的求解器之间完全一致；源与势都不施加非负约束。

L-BFGS 的核心只有：

```python
def objective(x):
    residual = (problem.data_active - problem.forward(x)) / problem.noise_active
    qx = problem.prior_matvec(x, lambda_source, lambda_potential)
    return 0.5 * (jnp.vdot(residual, residual) + jnp.vdot(x, qx))

value_grad = jax.jit(jax.value_and_grad(objective))
# 包装为 SciPy 所需的 float 和 NumPy gradient 后：
result = scipy.optimize.minimize(fun, x0, jac=True, method="L-BFGS-B")
```

没有设置 bounds，因此仍是无约束问题。新优化路径不调用原实现的解析梯度、Jacobian、LM 更新或 damping 控制。原 LM 仅作为对照运行；其公共 `fit` 仍自行计算最终 curvature/evidence，本文不比较或验证 evidence。

## 为什么增加固定坐标变换

默认对照三种参数化，求解的物理目标完全相同：

| 参数化 | 定义 | 额外工作 |
| --- | --- | --- |
| `raw` | \(x=x_0+z\) | 无预条件 |
| `whitened` | \(Q=LL^T,\ x=x_0+L^{-T}z\) | 固定先验白化 |
| `initial_whitened` | \(H_0=J_{\rm AD}(x_0)^TWJ_{\rm AD}(x_0)+Q\)，用其 Cholesky 白化 | **JAX 自动微分**计算一次初始 Jacobian |

后两者都是可逆坐标变换，不添加新 penalty；所有结果都转回物理变量后统一评估。
`initial_whitened` 的矩阵只准备一次，L-BFGS 循环不更新它。
可选的 `initial_diagonal` 仅使用上述初始矩阵的对角尺度，不分解完整 \(H_0\)。

这里显式构造小型 dense 矩阵来回答可行性问题。完整白化有平方存储和立方分解成本，尚不是大规模 operator 方案；这项成本不能从未来性能评估中省略。

## 首轮小网格实测

环境：CUDA GPU、float64、JAX 0.7.1、SciPy 1.17.1、NumPy 2.4.6、Python 3.11.14。
L-BFGS 最多 1500 步，`maxcor=30`、`maxls=50`、`gtol=1e-7`、`ftol=1e-15`；LM 使用库默认停止控制，`max_iter=160`。完整环境、脚本 SHA256、库提交和运行参数见 [environment.json](results_final/environment.json)。

五个案例、每个两个检查点中，AD 与现有解析分支梯度的相对 L2 误差最大为 **\(4.20\times10^{-14}\)**。三个步长的中心有限差分也已核对，每例至少一个步长的相对误差低于 \(10^{-7}\)，并记录两端未跨越插值单元或源边界。

下面是最终记录中的 `initial_whitened`。`g` 使用当前 LM 同样的物理尺度：
\(g=\|\nabla F/\sqrt{\operatorname{diag}(J^TWJ+Q)}\|_\infty\)。
端点 Jacobian 由 AD 独立计算，仅用于诊断，不反馈优化器。统一报告检查阈值为 \(g\le10^{-6}\)，不修改两个库各自的停止标准。

| 案例（数据数 / 联合未知数） | L-BFGS 最终 F | LM 最终 F | L-BFGS / LM 接受步数 | L-BFGS 的 g |
| --- | ---: | ---: | ---: | ---: |
| 常数源 ridge（16 / 37） | 13.714285714286 | 13.714285714286 | 2 / 3 | 7.7e-15 |
| 小型源—势耦合（16 / 37） | 0.446379594762 | 0.446379594762 | 35 / 16 | 9.1e-8 |
| SIS+shear nodal mock（84 / 122） | 0.0460958021481 | 0.0460958021481 | 7 / 6 | 1.0e-8 |
| 独立解析 subhalo mock（512 / 414） | 1054.417621605 | 1069.222769256 | 112 / 29 | 2.2e-3 |
| 独立解析 control mock（512 / 414） | 976.202941239 | 1052.836281384 | 31 / 12 | 3.7e-3 |

前三例均通过梯度、目标值及图像一致性检查：与 LM 的归一化目标差小于 \(10^{-10}\)，预测图像差的 RMS 小于 \(10^{-5}\sigma\)，端点满足统一分支梯度门限。ridge 的源另与解析解 \(8/7\) 核对。

**直接优化与预条件优化有明显区别。** nodal 上 `raw`、先验白化均耗尽 1500 步，前者 F=0.0483317961、后者 F=0.0461311876，均未通过梯度检查。固定初始白化后才在 7 步达到与 LM 一致的解。补充对角实验中 `initial_diagonal` 也达到相同目标，约 260 步、g=9.8e-8；见 [补充记录](results_initial_metric/nodal/results.json)。因此，完整初始曲率分解不是这个小例子达到同解的唯一途径。

**两个独立解析 mock 尚未通过充分收敛检查。** 它们包含 Gaussian PSF、同一噪声实现、2×2 子像素积分，图像由解析源和解析 Pseudo-Jaffe 偏折独立生成，再用 11×11 像素源和像素势拟合。L-BFGS 因目标变化小而返回 `success=True`，但 g 仍明显高于门限；LM 则分别 `stalled` 和 `linear_solve_failed`。L-BFGS 的目标更低只说明找到更低的点，不能证明已收敛到同一个解、全局最优或更稳定。

所有最终射线均在源 bbox 内，但这不排除内部双线性格点的梯度跳变。当前实验未隔离解析 mock 停滞的具体原因；条件数、内部插值折点和线搜索均需进一步区分。

subhalo 与 control 的势平滑强度分别为 1.6e7 和 1e11，源强度同为 6e-4，势幅度先验同为 1e4。这些是本次缩小网格实验的固定设置，不能用两者 F 的差解释检测显著性，也不能把本记录外推为现有 200×200 scientific 配置的结果。

## 验证与边界

- 已实际完成默认的五案例、三种 L-BFGS 参数化与 dense LM 对照；原始场、逐步历史、两类 penalty、梯度、状态和终止原因都已落盘。早期未经初始 metric 的记录保留在 `results/`、`results_analytic_initial/`，补充缩放实验在 `results_initial_metric/`。
- 已单独运行现有的三个回归：当前双线性梯度、ridge 解、独立非线性目标/evidence 参考，结果 **3 passed**。它们不构成对新路径 evidence 的验证。
- 计时分别记录首次 JIT、固定坐标准备、L-BFGS 求解及 LM 完整 fit；两种方法的范围不同，**不据此计算加速比**。SciPy 调度会传输 GPU 梯度到 CPU，也未代表纯 JAX 优化器性能。
- 所有初始源来自 truth 节点采样；这是算法验证，不是实际数据初始化、真实源恢复或低质量初值鲁棒性验证。
- 本次固定正则化强度，只验证 MAP 优化；没有把正则化强度混入裸 penalty 优化，没有拿 L-BFGS 的有限记忆逆 Hessian 代替 evidence 的 log-determinant。
- 未修改 `TinyLensGpu/`、现有 examples 或测试。根 `.gitignore` 仅为新目录的 `.py`/`.md` 增加白名单；数据与结果继续忽略。

这里留下的完整科学 Mock 验证已在后续实验中完成，具体配置、端点审计和观测源初值验证见 [后续报告](scientific_results.md)。纯 JAX Optax L-BFGS、源消元等方向的官方依据与区别见数学核对笔记，尚未混入本次实验。
