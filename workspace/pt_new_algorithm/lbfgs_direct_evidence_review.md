# 三强度 DIRECT + L-BFGS evidence 审查

范围：独立实验脚本；仅只读审查与 CPU 元数据核验，不运行拟合或 GPU 工作。科学目标是固定宏模型、固定离散网格下的经验贝叶斯点优化。

## 必须保持的目标

令 \(q=(s,p)\)，\(W=\operatorname{diag}(\sigma^{-2})\)，

\[
Q(\lambda)=\operatorname{blockdiag}\{\lambda_s R_s,\lambda_4D_4^TD_4+\lambda_0I\},\qquad
F(q;\lambda)=\tfrac12[(d-m(q))^TW(d-m(q))+q^TQq].
\]

每个候选先从同一 \(q_0\) 独立启动内层 L-BFGS，再在其终点 \(\hat q\) 计算

\[
H_{\rm GN}=J(\hat q)^TWJ(\hat q)+Q,\qquad
\log E_{\rm GN}=-F(\hat q)+\tfrac12[\log\det Q-\log\det H_{\rm GN}-\log\det C-N_d\log(2\pi)].
\]

这正是当前 iterative 的 `local_gauss_newton` 定义；可核查 [最终终点刷新与公式](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py:625)。它不是非线性模型的精确边际化，也不是使用真 Hessian 的 Laplace：真 Hessian 还包含 \(-\sum_i(d_i-m_i)\nabla^2m_i/\sigma_i^2\)，GN 省略此项。双线性插值还有分段边界，局部导数证书不能证明全局最优。

本地论文 [2510.07382v1.pdf，第 4、6–7 页](/home/xycao/workspace/PT_dev/paper_pdf/2510.07382v1.pdf) 的式 (4)、(17)–(18) 支持逐候选内层迭代后评价归一化 evidence 的流程；论文采用两个强度及 simplex，而本实验扩展为三个强度及 DIRECT。本地参考 [iterative.py:435](/home/xycao/workspace/PT_dev/lensing_potential_correction/potential_correction/iterative.py:435) 也使用终点 GN 曲率，但其 `jitter=1e-6` 属于该实现细节，不应移入新实验来改变既有先验。技能列出的 `staf2253.pdf` 在本机未提供，不据此虚构引文。

## 先验、缓存和内层接纳

- \(\lambda_0\) 同时改变 penalty、初始白化和两个 determinant；\(\log\det Q_p\) 必须来自完整 \(\lambda_4D_4^TD_4+\lambda_0I\)，不能用 \(N_p\log\lambda_4\) 代替。原有 [all-node amplitude prior](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/utils/inversion/potential_regularization.py:66) 对没有完整四阶窗口的节点同样有效。
- 当前 source second-order prior 的近边界一阶与外边界零阶行固定住零模，所以当前 \(R_s\) 是 proper prior；[边界构造](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/utils/inversion/regularization.py:1449) 和 [完整 block prior](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/_potential_correction_common.py:151) 是依据。可复用 \(\log\det R_s\)，但须加 \(N_s\log\lambda_s\)。不得把一般无边界 curvature prior 都视为 SPD。
- 固定 geometry、data/noise、dtype、先验基矩阵和同一个初始 \(q_0\) 时，可缓存 \(J_w(q_0)^TJ_w(q_0)\)；候选仍各自构造 \(Q,H_0\) 和 Cholesky，且从零白化坐标启动。终点 Jacobian、\(H_{\rm GN}\) 与 determinant 不可用初始 metric 或 L-BFGS `hess_inv` 替代。
- 三个强度应作为稳定 JIT 的动态数组参数，不能每候选新建捕获强度的函数；[JAX JIT/cache 说明](https://docs.jax.dev/en/latest/jit-compilation.html) 支持该做法。缓存分数须保留完整三维坐标和运行协议，不能把两组不同 \(\lambda_0\) 当成同一候选。
- 接纳规则应在搜索前声明，只使用优化与数值证书，不用 truth recovery 筛选。`native success`、有限值、SPD 和 \(\tfrac12g^TH_{\rm GN}^{-1}g\le10^{-3}\) 是可审计的工程规则；最后一项只是 GN 二次模型的预测可降幅，不是真实 gap 上界，也不能证明 evidence 误差小于 \(10^{-3}\)。旧 physical gamma 单独记录。若用 \(D^{-1}HD^{-1}=LL^T\) 分解，证书应为 \(\tfrac12\|L^{-1}D^{-1}g\|^2\)。
- 失败、非有限或不正定候选保留原生原因，返回无 evidence/正无穷外层代价；不能用另一个内层求解器悄悄救回。宽边界下失败区域应报告，因为最优值只是在本协议可解区域内寻找。

## 外层与解释边界

沿用旧实验的 \(\log_{10}\lambda_s\in[-8,-1]\)、\(\log_{10}\lambda_4\in[1,11]\)、\(\log_{10}\lambda_0\in[0,8]\)，其来源是 [旧三强度 runner](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/.scratch/potential-correction/experiments/optimize_single_step_three_strengths.py:138)。这些是实验边界，不是由本次数据证明的物理范围。最大化 evidence 不因改用 log 坐标额外添加 Jacobian；超参数先验或积分是另一项推断任务。

同一数据的 baseline→selected evidence 改善可以报告；subhalo 和 matched-control 是不同数据，二者各自最佳 evidence 的差不能解释为 detection Bayes factor。三个强度取最大值不等于对其边际化，也不提供超参数后验不确定度。

[SciPy DIRECT 文档](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.optimize.direct.html) 将 `maxfun` 定义为近似预算，`len_tol` 是归一化最优矩形的尺度条件。应报告实际调用数、失败数、原生 `success/message` 与最优点距边界；预算耗尽仍可得到当前最佳有限候选，但不能称已证明全局最大值。外层不需要 evidence 的梯度；[L-BFGS-B 文档](https://docs.scipy.org/doc/scipy-1.17.0/reference/optimize.minimize-lbfgsb.html) 的 `ftol` 和 `gtol` 是内层停止条件，不能混作 DIRECT 收敛证书。

主实验采用观测 source-only MAP：在 \(p=0\)、固定初始化强度 \(\lambda_s=0.0006\) 下求 signed source，随后冻结所得 \(q_0\)，不随候选强度重新求初态。此时 source response 线性，可在 zero source 用 AD 构造 source Gram；初始 joint Gram 则须在实际冻结的 \(q_0\) 构造。应保存初态和 `source_truth_used=False` 的 provenance；宏模型依然是标准 mock 的固定真值。旧 truth-node 初始化仅是明示的可选 CLI 模式。固定初始 Gram 与 source-only 初始化的一次性准备应单列；本次每候选速度不能直接等同于此前每 fit 重建完整初始 metric 的约 1 秒基准。

## 新脚本与结果

只读审查 [run_lbfgs_direct.py](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/run_lbfgs_direct.py) 未发现阻塞的数学/API 问题：动态 prior action 与 dense Q 同时接收三强度；source determinant 使用 \(N_s\log\lambda_s+\log\det R_s\)，potential determinant 使用完整矩阵；\(H_0=D_0L_0L_0^TD_0\) 对应的 decode 是 \(q=q_0+D_0^{-1}L_0^{-T}z\)。终点重新构造 AD Gram 和 undamped H，stationarity proxy 的尺度顺序正确；独立 analytical audit 用新构造实例，避免 JIT 捕获旧 \(\lambda_0\)。所有候选从同一 `q0`、白化坐标零开始，无内层 fallback 或 truth 筛选。

CPU 核验 [preflight_v2](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_direct_preflight_v2/protocol.json) 的两个 baseline 与两个 repeat 均通过数值 gate、原生成功。source-only 线性方程相对残差分别为 \(7.79\times10^{-16}\)、\(7.26\times10^{-16}\)，`source_truth_used=False`；3357 个联合参数。两 worker 的源代码 SHA 与归档 snapshot 一致（`09fa5be9ab392d225…`）。

| preflight | baseline logE | 重跑 ΔlogE | 同态 AD/analytic logE 差 | 梯度差 / GN scale |
|---|---:|---:|---:|---:|
| subhalo | 69037.2566988224 | \(8.61\times10^{-5}\) | \(1.16\times10^{-10}\) | \(6.80\times10^{-14}\) |
| control | 69487.8377649831 | \(1.22\times10^{-8}\) | \(1.00\times10^{-7}\) | \(3.49\times10^{-13}\) |

这同时显示目标有可测重跑波动，不能把最后若干小数当搜索精度。`Engine.preparation_seconds` 包含初始数组输出，候选 `total_seconds` 到 stationarity/evidence 为止，导出与 independent analytical audit 在外部；应按此口径解释耗时。

## 正式 full200 结果核验

[正式归档](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_direct_full_v1/runner_at_measurement.py) 为 `f6e2ad418691e85d…`，与 worker 记录一致，9 个依赖 SHA 已对当前文件核验。正式版本在搜索前冻结 baseline 重跑容差 \(|\Delta\log E|\le0.05\)；selected 重跑仅作一致性诊断，不替换选中的候选。导出中的 \(\lambda_0\)、penalty 和 objective 元数据已全部纠正为当前候选值。

subhalo 的 [summary.json](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_direct_full_v1/subhalo/summary.json) 和全部 306 条 trace 已做 CPU 审查：305 次 DIRECT 调用加独立 baseline，267 个 accepted，17 个原生失败、22 个 GN proxy 超限；306 个端点文件齐全，失败 evidence 全为 null。通过行均为原生成功且满足冻结 proxy gate，完整 evidence 公式重算误差为零，所有行初态 hash 相同。输入 image/noise 与原 validated-reference 逐元素一致；初始 potential 全零，source 与 truth 的相对 L2 差 9.63%，符合 observation-only 初始化来源。

DIRECT 原生 `success=False,status=1`，实际 `nfev=305,nit=29`，耗尽近似 `maxfun=300` 预算。搜索耗时 536.768 s（含候选写盘/调度），记录的 worker 时段为 558.096 s。当前最佳为真实搜索候选 #128：

\[
(\lambda_s,\lambda_4,\lambda_0)=(7.490065\times10^{-5},\ 8.273618\times10^5,\ 2.679067\times10^4),
\quad\log E=69538.8836167,\quad\Delta\log E=501.6268249.
\]

三个坐标均远离所设边界；所选点 proxy 为 \(8.44\times10^{-6}\)，旧 gamma 也通过。两次相同初态重跑均 accepted，\(\Delta\log E=+0.00244510,-0.00178875\)，在预设容差内；baseline repeat accepted 但旧 gamma 未通过，这一标志保持原样。独立同态 AD/analytic selected evidence 差 \(5.82\times10^{-11}\)，梯度差/GN scale 为 \(1.19\times10^{-13}\)。CPU 独立按 source 边界 stencil、1592 个完整 D4 窗口与全节点 amplitude 项重算 penalty，baseline/selected 的 F 差分别为 0、\(1.82\times10^{-12}\)。

物理诊断显示权衡，不能概括成整体恢复改善：峰值保持在 \((0.75,0.075)\)，\(\kappa_{\max}\) 从 0.3035766 升至 0.4767008，更接近离散 truth 0.631435；但全 core RMSE 从 0.0267763 升至 0.0382699，local RMSE 从 0.0582775 升至 0.0619801。source 相对 L2 误差仅从 0.0702874 降至 0.0698025，残差 RMS/noise 从 1.0018375 降至 0.9921248。这些值已由保存数组重新计算，未反向参与候选选择。

control 的 [summary.json](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_direct_full_v1/control/summary.json) 同样经过全部 306 行和保存数组审查：291 个 accepted，4 个原生失败、11 个 proxy 超限。DIRECT `success=False,status=1,nfev=305,nit=36`，同样因预算停止；搜索 201.200 s，记录的 worker 时段 219.057 s。最佳候选 #94 为

\[
(\lambda_s,\lambda_4,\lambda_0)=(8.552612\times10^{-5},\ 8.675042\times10^{10},\ 8.925186\times10^7),
\quad\log E=69808.5001435,\quad\Delta\log E=320.6623785.
\]

\(\lambda_4,\lambda_0\) 距各自上界仅 0.06173、0.04938 dex，两者均触发预设 1% 范围的近边界标志；此结果不足以确定无限制搜索的最优强度。所选点 native `gtol` 成功，proxy 为 \(4.95\times10^{-16}\)，旧 gamma 通过。两重跑 accepted，evidence 差仅 \(\mp1.46\times10^{-11}\)；独立 analytical evidence 差 \(2.04\times10^{-10}\)，梯度差/GN scale 为 \(1.65\times10^{-13}\)。NumPy stencil 重算 baseline/selected F 均只差 \(1.82\times10^{-12}\)。control 初始 source 与 truth 相对 L2 差为 5.10%，potential 全零；原数据/noise/axes/nodes 与 validated-reference 相同，且两原始案例的 noise realization 逐元素相同。

control 的 \(\kappa\) core RMS 从 \(5.32938\times10^{-4}\) 降至 \(7.63851\times10^{-5}\)，残差 RMS/noise 从 0.9958285 降至 0.9936119；source truth 相对 L2 误差却从 0.0517100 升至 0.0551242。因此控制场更接近零，不表示 source 也更准确。

全部 612 个 trace 候选中，有 558 个按新规则 accepted；旧 gamma 仅分别有 175/267、245/291 个 accepted 候选通过，二者不能混称。失败不获 evidence、无重试/替换，best 确为所有 accepted 行的最大分数。正式两例与各自 baseline 的证据改善远大于已观察重跑波动，足以支持这个外层实验已跑通；它仍受预算、失败区域、局部 GN 近似与 control 上界约束，不能宣称全局收敛、超参数 posterior、检测显著性或整体真值误差改善。

计时补充：`total_process_seconds` 从 worker 完成 imports 后、生成 mock 前开始，包含 mock/geometry、初始化、准备、搜索、审计及导出，排除 Python 解释器与 imports 的启动；它不是外部命令的总 wall time。每例只有 seed=17 的一组数据，重复只检验同一输入的数值重现性。
