# J0946 迭代 potential correction 审查

本笔记仅审查真实 Jackpot/J0946 数据、M3 交接和新实验的数学/实现；不运行宏模型、科学拟合或 GPU 工作，仅作小型 NumPy/API 核验。目标为固定现有 M3 宏模型后的非参数 potential correction，不是 mock，也不以参数化 subhalo 替代。

## 交接数据与冻结项

只读核验 [原 m3_handoff.json](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/slacs0946_potential_correction/macro/m3_handoff.json) 和 NPZ，SHA256 与记录的 `d09a1102a64f74dc…` 一致。M3 代表点是保留样本中 likelihood 最大的样本 #21342，不是 posterior median；[export_m3_bundle](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/slacs0946_potential_correction/macro_model.py:2060) 是来源。已有 smooth fit 不应重跑。

- 原生图像 201×201、dpix=0.05 arcsec、nsub=4，15×15 PSF 和为 1。`mask=True` 排除，共保留 2293 个像素；同一真实数据和 PSF/mask 应直接复制、校验。其中 8 个保留像素的 noise=\(10^8\)，应沿用而非自动删除，Gaussian normalization 仍包含全部 2293 个像素。
- source 为 100×100，signed intensities，row-major；x/y 轴均递增。bbox 按 `(xmin,xmax,ymin,ymax)` 为 `[-0.5641691089,0.4358308911,-0.6536976099,0.3463023603]` arcsec。原 handoff 全部数值数组为 Float32，推广到 Float64不代表重新拟合宏参数；重建坐标与网格的舍入/边界差异应记录。
- NPZ 的 `lens_subtracted_image` 在 Float32 下严格等于 `image_data-lens_light_model`。新问题应直接使用这一保存的 datum，forward 的 `lens_light=[]`，不再次减光，也不悄悄用 Float64重新相减替换原数据。noise 沿用保存值；这是条件于固定透镜光的推断，没有传播透镜光拟合不确定度。[旧 PT load_problem](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/slacs0946_potential_correction/pt.py:90) 同样如此。
- M1 template、M2 的 rho=5.530494995 和 adaptive source scale 固定。保存的 10000 个 scale 均有限正值，范围 `[1,252.2687988]`；禁止随迭代 source 再估计权重。first-order source prior 的边权、边界锚定、penalty 和 determinant 必须使用同一 scale。[原 source matvec](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/utils/inversion/regularization.py:325) 的外边界 zero-order 行使正权图的 prior proper/SPD，不能当作无边界的奇异 Laplacian。
- M3 source 是观测拟合的 reference/可选初始化，不能标为 truth source；实际 frozen 实验采用下述 conditional-source 初态，potential 均从 0 开始。`IterativePotentialCorrection` 当前构造器硬编码 truth provenance，实验实例需显式覆盖。它暂不接收 `source_scale` 参数，若实例赋值则必须在任何 prior 构造或首次 JIT tracing 前完成，并对独立 audit 实例同样设置。

旧 single-step 固定 M3 source 的空间梯度，但仍联合求解 source 与 potential；新 iterative 必须刷新当前 source、当前 potential 所对应的 corrected rays/双线性梯度，source 系数仍参与内层联合优化。冻结的项是 macro、网格、mask、PSF、adaptive map，不是新 source 的空间梯度。[iterative linearize](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py:243) 给出该约定。

## 规模与数值方法

[现有单步结果](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/slacs0946_potential_correction/potential_qr_v2/maximum_evidence/result.json) 的 geometry 为 \(N_d=2293,N_s=10000,N_p=1134,N_{\rm core}=802\)，potential spacing=0.1 arcsec。Float64独立构造后的实际计数须再核对。仅 \(11134^2\) 的 Float64矩阵约 992 MB，直接复用 mock 的 full joint Gram/Q/H/Cholesky 会放大显存和每候选分解成本；不得为省资源静默改成较粗 source 网格。

可保持完整 GN target 而利用 \(N_d\ll N_q\)：AD 按 data 行构造 \(J_w\)，其约 204 MB；若 \(Q=AA^T\)，令 \(V=A^{-1}J_w^T\)，则

\[
H_{\rm GN}=Q+J_w^TJ_w,\qquad
\log\det H_{\rm GN}-\log\det Q=\log\det(I+V^TV).
\]

这是完整 determinant lemma，保留全部 source–potential cross terms；[旧 PT 的 data-space/augmented-QR 实现](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/slacs0946_potential_correction/pt.py:230) 提供本地依据。若采用该形式，终点 \(J_w\) 和 prior root 都必须匹配当前 fields/三强度；不能复用 single-step 的冻结响应作为 iterative endpoint curvature。

如仍要求初始 GN 白化，可用 prior root 与低秩 \((I+VV^T)^{-1/2}\) 的作用避免每候选 full-joint 分解；若改用仅 prior/diagonal 白化，也应明示预条件算法变化、独立核对坐标梯度，物理目标保持一致。source prior 的基矩阵/基 factor 可固定准备，候选 \(\lambda_s\) 只缩放。不能通过 jitter 或改 amplitude prior 来掩盖分解失败。

## Evidence 与外层规则

内层仍优化同一个 signed 联合目标

\[
F(q;\lambda)=\tfrac12[\|W^{1/2}(d-m(q))\|^2+q^TQ(\lambda)q],\quad
Q=\operatorname{blockdiag}(\lambda_sR_s,\lambda_4D_4^TD_4+\lambda_0I).
\]

\(\lambda_0\) 进入所有 potential 节点、penalty 和归一化；evidence 为终点刷新、不加 damping 的 local-GN 高斯近似，区别于真 Hessian Laplace 和精确非线性边际化。本地 [论文 §1.4](/home/xycao/workspace/PT_dev/paper_pdf/2510.07382v1.pdf) 与 [现有 iterative evidence](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py:625) 是流程依据。

每个候选按同一冻结的数据初始化规则冷启动/零 potential，不继承其他候选解；外层 DIRECT 后的 local refinement 同时优化三个 log-strengths。local 用显式有限预算的 bounded Powell，避免对有内层/分段曲率波动的 evidence 用默认微小有限差分梯度。新 stopping 数值 gate 应在搜索前声明，不以重建图像外观决定；native success、失败原因、GN proxy、旧 gamma 和外层退出原因分别保存，失败不获 evidence。

旧 single-step 的数据依赖 bounds 可作为明确声明的实验范围，但优化最高 evidence 是 empirical-Bayes 点估计，不是原 Nautilus hyperposterior。M3 nested logZ 含位置惩罚和 operator determinant，不能作 PT Bayes-factor baseline；single-step conditional-linear evidence 与 iterative local-GN evidence 的不同近似也必须标注。峰值不自动等于检测显著性或质量测量，不能凭图像观感调 prior/选择解。

## 新入口与结果核验

已完整只读审查 [pt.py](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/iteration/pt.py) 与 [lbfgs.py](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/iteration/lbfgs.py)。loader 仅读取复制的 M3，不 import 或运行宏模型脚本；保存的 native Float32 image queries 先生成后推广，source_scale 在首次 tracing 前设置，fresh analytical audit 也保留同一 adaptive prior。实际 Float64 geometry 为 10000 source、1134 potential、802 core、2293 data，与旧计数一致。当前 M3 初态重渲染与保存 Float32 model 的 selected-pixel 最大差为 \(1.43205\times10^{-5}\) 强度单位；这是记录的浮点实现差异，不称 bitwise 同一 forward。

低秩白化顺序正确：\(H_0=A(I+VV^T)A^T\)，\(T=A^{-T}B\)，\(B=(I+VV^T)^{-1/2}\)。以 \(I+V^TV=U\operatorname{diag}(c)U^T\) 分解，代码使用稳定系数 \(1/[\sqrt c(1+\sqrt c)]\)，没有零特征值相消/除零；source 的 \(\sqrt{\lambda_s}\) 尺度在 prior root、decode 和 endpoint response 一致。其正范数 stationarity proxy 也正确：令 \(u=A^{-1}g,t=(I+V^TV)^{-1}V^Tu,r=u-Vt\)，则 \(\frac12(\|r\|^2+\|t\|^2)=\frac12g^TH^{-1}g\)。纯 NumPy 两个极小例（分别 \(N_d<N_q\)、\(N_d>N_q\)）与独立 dense inverse/determinant 比较，白化误差不超过 \(7.11\times10^{-15}\)，proxy 差不超过 \(2.22\times10^{-16}\)，determinant lemma 差不超过 \(1.78\times10^{-15}\)；未 import JAX 或运行拟合。

代码审查发现并反馈的两处正式 workflow 问题已修正：installed SciPy DIRECT 要求 `bounds.tolist()`/Bounds，不能直接传 NumPy ndarray；selected audit 不再更新/覆盖 retained best，先冻结原分数后作双向绝对差比较。经验 bounds 的 median 也已改为仅使用 informative positive potential-Gram entries，与旧规则一致。Float64 重新计算的范围可与旧 Float32 数值不同，需以新 protocol 为准；新版本加入源代码 snapshot/hashes。

首个 [preflight](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/iteration/preflight/failure.json) 的白化 quadratic error 为 \(7.77\times10^{-16}\)，endpoint augmented-QR 与 Cholesky determinant-ratio 差 \(5.46\times10^{-12}\)，同态 AD/analytic F 差为 0、梯度相对差 \(4.32\times10^{-13}\)，证明接口和代数通过。但 native L-BFGS 在 F=506.8662431、131 iterations 后以 rel-F 成功退出，GN proxy=0.1049834，超过预注册 \(10^{-3}\)，因此 `accepted=False,log_evidence=None`。其 diagnostic logE 不可用于正式选择。

修正 positive-median bounds 后的 `preflight_v2` 达到 3000 iterations，仍未通过数值 gate。已有 single-step 强度附近的 [strong_prior_validation](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/iteration/strong_prior_validation/evaluations.jsonl) 也只通过代数/AD 核验：native rel-F 成功、2617 iterations，proxy=0.0106927，仍然 rejected。不能把程序退出码或原生 success 单独当成可信收敛。

[endpoint diagnosis](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/iteration/preflight/endpoint_diagnosis.json) 记录当前 rays 距 hard bbox 仅 \(1.01\times10^{-10}\) arcsec。沿 AD 下降方向，白化 step=\(10^{-5}\) 时预测 \(\Delta F=-3.89\times10^{-6}\)，实际 \(+0.03129\)；step=\(10^{-8}\) 时实际下降与 AD 一致。重置 L-BFGS memory/ftol 诊断没有解开这一停滞。其源代码依据是 [mapping.py 的 valid/zero-outside 分支](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/utils/lensing/mapping.py:57)。该证据支持当前目标的 hard-bbox 非光滑/不连续障碍，不能通过放松 gate 宣称成功；也不能把 GN proxy 当真实剩余 gap。边界 source 非零时 hard cutoff 不局部 Lipschitz，单条有限差分 profile 不构成 Clarke stationarity 证书。

可进一步严格区分 source 与 potential 子问题：固定 p 时目标对 s 为精确二次式，因此 \(\frac12g_s^T(\lambda_sR_s+J_s^TWJ_s)^{-1}g_s\) 是 source-only 的精确可降幅。若该项小而 potential proxy 大，只说明 potential 的 nonsmooth stationarity 未解决。按候选 \(\lambda_s\) 在 p=0 作 observation-only source MAP 可作为明示的新初始化协议，但会改变初始 source，故 \(J_{0p}\) 也必须刷新，或把旧 M3 metric 明示为 fixed precondition；不能同时称它是新起点的精确 GN 白化。这不改变 physical target，也不解决 hard-bbox 本身。

## 冻结的 DIRECT + Powell 实验

[direct_powell 归档](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/iteration/direct_powell/protocol.json) 使用新的 `conditional_source` 初始化，DIRECT `maxfun=60`、bounded Powell `maxfev=60`，inner 最多 3000 iterations，proxy 门限仍为 \(10^{-3}\)。每个候选在 p=0 解同一观测数据的 source-only MAP，随后联合 source/potential L-BFGS；没有重拟合 smooth macro、导入单步 potential 解或改变 prior。M3 source 用于固定 reference GN precondition，与实际候选初态明确分开；白化在 reference 点仍精确，实际起点处不声称精确白化。

source 初始化的 data-space 公式正确。设 \(R_s=D_sC_sC_s^TD_s\)、\(S_0=C_s^{-1}D_s^{-1}J_{s,w}^T\)，则

\[
s_0(\lambda_s)=\lambda_s^{-1}D_s^{-1}C_s^{-T}S_0
(I+S_0^TS_0/\lambda_s)^{-1}W^{1/2}d.
\]

因为 p=0 时 source response 线性，缓存的 \(J_s,S_0\) 不依赖 M3 source intensity，复用合法。纯 NumPy 六个极小例涵盖 \(N_s\gtrless N_d\)、\(\lambda_s=10^{-3},0.4,10^3\)，与独立 normal-equation 直接解的相对差不超过 \(3.01\times10^{-13}\)、方程相对残差不超过 \(3.56\times10^{-13}\)。

冻结的 strong-prior validation anchor \(\log_{10}\lambda=(-3.4849016084,7.5,6.5)\) 已数值 accepted：native rel-F success、23 iterations，proxy \(3.17\times10^{-8}\)，logE=6211.0675975；algebra/analytic audits 均通过。这证明原 physical target 存在可认证区域，不代表更弱正则化的失败区域已经解决。anchor 在外层开始前声明，不以重建外观选择。

provenance 注意：运行时原始 protocol/export 的 `initialization.source_initialization` 与 NPZ `source_initial` 描述保存的 M3 reference；实际候选起点以 `cold_start`、CLI 参数及每行 `source_initialization="conditional_source"` 为准，不能向读者称所有候选从同一 M3 source 开始。最终 export 已按下述可追踪记录修正，原版另行归档。

首个已认证 anchor 的保存数组也经过独立 NumPy 检查：adaptive first-order 边权与原无权边界锚定、1788 个完整 D4 窗口、全部 1134 节点的 amplitude 项重算 penalty，误差 \(2.27\times10^{-13}\)，F 误差 \(1.14\times10^{-13}\)。datum/noise/mask/source_scale/M3 reference source 与原 handoff 逐元素一致；复制的 macro_model.py 与原文件 SHA 一致，10 个记录的 snapshot SHA 全部核对。该 anchor 的 κ peak=0.00739545、802 个 finite derivative 节点、残差 RMS/noise=0.740971，仅表示强 prior 条件下的认证重建，不是检测结论。

DIRECT 的 `maxfun` 是软预算：[installed wrapper](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/scipy/optimize/_direct_py.py:257) 将原值传给 native，没有改成至少 100；[随包测试](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/scipy/optimize/tests/test_direct.py:169) 明确断言 `nfev>=maxfun`。极小纯 CPU 三维 API 探针中，unbiased DIRECT 的 sphere `maxfun=10/20/60` 实际为 `21/21/101` 次，constant/mostly-invalid 为 `15/27/189` 次，均预算退出 status=1。故大量失败同分时实际调用可明显超过名义 60，应按实际 nfev/wall 报告；这些 toy 次数不是对真实 campaign 次数的推断。

官方 [SciPy v1.17.1 DIRect.c](https://github.com/scipy/scipy/blob/v1.17.1/scipy/optimize/_direct/DIRect.c#L488) 进一步确认：原版会追加同层同分矩形，本轮选中矩形的采样/划分之后才检查 `numfunc>maxf`。源码另含未找到 feasible point 时调整预算的分支；本次是否进入该分支未验证，不作归因。

## 正式 raw cohort：有认证试点，选中点重复证书失败

[原始 trace](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/iteration/direct_powell/evaluations.jsonl) 共 254 次科学拟合记录、253 个 distinct hyperpoints：1 baseline、193 DIRECT 新点、59 Powell 新点、1 强制 selected audit。36 行 accepted（1 baseline、10 DIRECT、25 Powell）；218 行 rejected，分别为 75 次 maxiter 原生失败、143 次原生 rel-F success 但独立 proxy 未达门限；无 factor exception。所有行的完整 evidence/diagnostic evidence 均用 F、determinant ratio 与 2293 个像素的 noise normalization 在 CPU 重算，误差为零；未获接纳的 evidence 全为 null。

原程序在 selected-repeat 检查失败后退出，未及时序列化 outer 结果；因此按授权另外保存 [replayed_outer_status.json](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/iteration/direct_powell/replayed_outer_status.json)。它仅用已保存分数做 CPU replay，不调用任何新模型；未知坐标即报错，保留原 round(x,12) cache 规则。DIRECT 的 193 个查询逐点一致；Powell 的 59 个新点顺序一致，另有 1 次既有点 cache 命中，查询序列 SHA 和 trace SHA 已保存。

| 外层 | 原生退出 | 实际调用/迭代 | 返回点与保留最佳的关系 |
|---|---|---|---|
| DIRECT，名义 maxfun=60 | success=False，status=1，预算退出 | nfev=193，nit=5 | 返回 logE=5970.83778，未胜过外部 validation anchor 6211.06760 |
| bounded Powell，maxfev=60 | success=False，status=1，预算退出 | nfev=60，nit=1 | 原生最终点代价=\(10^{30}\)，是 rejected 点；另行保留它曾访问的最佳 accepted trial |

保留 trial #236（Powell）为

\[
(\lambda_s,\lambda_4,\lambda_0)=(1.0843505\times10^{-4},\ 7.3782954\times10^5,\ 1.5705035\times10^6),
\quad\log E=6240.4644678,\quad\Delta\log E_{\rm anchor}=29.3968704.
\]

它不靠近设定 bounds，单次 native 成功，proxy=0.000667608，F=629.1329530。保存 q 与 `candidate_0236.npz` 逐元素一致。由保存数组独立重算 adaptive source/D4/amplitude penalty 后，F 差 \(1.14\times10^{-13}\)，penalty 差 \(5.68\times10^{-14}\)。field 的正峰 0.04366474 位于 \((-0.7,1.0)\) arcsec，负峰 −0.07191279，core κ RMS=0.00995950，残差 RMS/noise=0.662764；这些只是该条件重建的诊断，没有注入 truth 可供恢复误差评价。

**关键限制是 selected cold-start repeat #253 未通过同一门限。** 该次仍是 native rel-F success、代数/QR/analytic audits 通过，但 proxy=0.001134934 >0.001，因此 `accepted=False,log_evidence=None`；diagnostic logE=6240.4603123，与原保留点差 −0.00415548。分数差虽小于预设 0.01 重现容差，stationarity gate 独立失败，不能称验证成功。原 best 未被这次重复替换，也未降低门限或升级较高 diagnostic 分数的失败点。

因此现有产物支持“新 L-BFGS + DIRECT + Powell 流程已执行、完整归一化与导数核验正确，并找到单次可认证候选”；不支持“已获得可重复认证的最优解”。搜索只覆盖部分可认证区域，未解决 hard-bbox 障碍、全局最优性、物理 subhalo 显著性或质量推断。后续任何额外验证都应逐次并列保留，不得以新增通过记录抹去 #253 的失败。

## 最终补充验证、状态与 provenance

[三次尝试记录](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/iteration/direct_powell/repeat_comparison.json) 已完整核验，门限始终为 \(10^{-3}\)：

| 同一参数/数据的尝试 | proxy | 门限结果 | diagnostic logE |
|---|---:|---|---:|
| 搜索内 trial #236 | 0.000667608 | 通过 | 6240.464467834 |
| 同进程冷启动 #253 | 0.001134934 | **未通过** | 6240.460312349 |
| 新进程独立验证 | 0.000935254 | 通过 | 6240.467861399 |

三次 native 均 success；两次专门复跑的 whitening/QR/analytic audits 通过。CPU 独立以原 potential 系数和中央差分重算 κ，复现相对 trial 的 joint 相对差 \(1.3560\times10^{-5},1.5205\times10^{-5}\)，κ 最大差 \(4.2545\times10^{-6},4.4674\times10^{-6}\)；峰位置均为 \((-0.7,1.0)\)。物理场接近与严格 gate 未全部通过这两个结论同时成立，不能由第三次通过抹去第二次失败，也不能用三次同一数据运算推断跨数据稳定性。

最终 [summary.json](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/iteration/direct_powell/summary.json) 和 [failure.json](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/iteration/direct_powell/failure.json) 明确保留 `status=repeat_gate_failed`、`selected_repeat_passed=false`、`all_attempts_passed=false`，正确区分 254 个 fit 记录和 253 个 hyperpoints。原进程 monotonic 总计时因预定失败未保存，`total_wall_seconds=null`；约 52.54 min 是 trace/protocol 时间戳估计，不能改称精确总耗时。

另有两项数值检查分别保留，未混入正式 254 行：早期 `conditional_source_validation` 在中点达到 3000 iterations、proxy=0.03620，仍 rejected；最终 [默认入口验证](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/iteration/final_default_validation/summary.json) 在 λ4/λ0 上界 anchor 24 iterations、proxy \(4.33\times10^{-7}\)，accepted 且代数/analytic audits 通过。它只证明当前默认数值检查点可运行，不证明已执行默认大预算搜索。

[export_provenance.json](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/iteration/direct_powell/maximum_evidence/export_provenance.json) 的原/新文件 SHA、immutable trace SHA 和 replay trace/protocol SHA 均已独立核对。`original_export/` 与修正后的 q、source、potential、κ 数组逐元素相同，evidence 完全相同；修正仅补明确切初始化与未解决的 repeat 状态。新 `source_initial` 等于独立进程捕获的 conditional MAP，`source_reference_M3` 等于原 handoff source；二者相对 L2 差 10.04%，对应初始 F 复核差小于 \(2.3\times10^{-13}\)。没有以新增通过的场替换原选中场。

37 个原始文件及其复制品（每套 154796984 bytes）已逐文件重新计算 SHA，全部与 manifest 一致；原 M3/smooth 结果未被改写或重跑。正式 snapshot 与两个最终验证各自记录的 10 个 source SHA 均匹配。最终 [results.md](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/iteration/results.md) 的计数、状态、场诊断、重复失败和 evidence 限制与这些原始记录一致，未发现阻塞的数学或 provenance 问题；方法在这个真实数据上的完整稳定优化仍未过关。
