# 原始 200×200 mock：配置、对照与验收核对

2026-11-03。此文记录原始基线和新优化器实验的科学验收。基线与新 L-BFGS 数字在各自标明的表中，均链接到具体输出。本核对只做代码阅读和 CPU 数组/metadata 检查，不运行 GPU 任务。

## 必须复用的原始问题

配置来源：[iterative_scientific_configuration.json](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/examples/potential_correction/iterative_scientific_configuration.json:1)；装配来源：[run_iterative_scientific](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/examples/potential_correction/iterative_scientific.py:163)。

| 项目 | 原始配置 |
| --- | --- |
| 图像/积分/精度 | 200×200，0.012 arcsec/pixel，生成与反演 nsub=2，float64 |
| 宏模型 | SIS theta_E=0.8，中心零，shear=(0.05,0.02)，固定 |
| 注入 | 解析 Pseudo-Jaffe，b_sub=0.04，位置(0.75,0.1) arcsec |
| control | 注入强度零，与 subhalo 共用同一噪声抽样 |
| 噪声/PSF/mask | sigma=0.01，seed=17；27×27 Gaussian，sigma=0.05 arcsec；annulus 0.45–1.1 arcsec |
| 解析源 | Gaussian widths 0.16/0.12 arcsec，clump_scale=0.7；其他参数保留原 JSON |
| 源网格 | 49×49，[-0.4,0.4]²，second-order prior |
| 势网格 | spacing=0.075 arcsec；724 core、956 support unknowns |
| 实际规模 | 22,016 个有效数据，88,064 条 query subpixel rays，3,357 个联合变量 |
| 固定强度 | lambda_source=0.0006；lambda_potential=subhalo 1.6e7/control 1e11；lambda_amplitude=1e4 |

生成图像使用解析源与解析偏折，反演使用像素源和离散势，因而这比同网格生成的 nodal fixture 更严格。原始装配固定几何和宏模型，允许源/势系数为负。不得为了简化原型改成32或64像素、换 PSF、缩小源网格、冻结初始插值、加入非负或重新选择正则化强度后仍称“原始配置”。[mock 生成器](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/examples/potential_correction/mock.py:204)。

## 已保存的原始 LM 结果

当前 checkout 内这些文件确实存在，可以直接读，避免重跑 LM：

- [subhalo metadata](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/.scratch/potential-correction/output/iterative-scientific/validated-reference/subhalo/dense/result.json) / [arrays](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/.scratch/potential-correction/output/iterative-scientific/validated-reference/subhalo/dense/reconstruction.npz)。
- [control metadata](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/.scratch/potential-correction/output/iterative-scientific/validated-reference/control/dense/result.json) / [arrays](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/.scratch/potential-correction/output/iterative-scientific/validated-reference/control/dense/reconstruction.npz)。

`load_result` 可以恢复公共结果，NPZ 也可以直接作为 NumPy 只读对照；保存的 metadata 包括完整 mock 配置、强度、源盒、几何、初始化和停止控制。[载入代码](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/examples/potential_correction/result_products.py:325)。

| 原始 dense LM 指标 | Subhalo | Control |
| --- | ---: | ---: |
| 目标 F | 11734.826513785734 | 11404.44365565745 |
| chi² | 22096.977698633775 | 21832.70162811225 |
| 未乘1/2的总 penalty | 1372.6753289376954 | 976.1856832026506 |
| 最终 GN 对角缩放梯度 | 4.2900044868e-4 | 2.5647942837e-5 |
| 接受步数 | 46 | 18 |
| 非线性 residual RMS/noise | 1.0018373768 | 0.9958284545 |
| 源相对 L2 | 0.0702923145 | 0.0517101271 |
| 所有 core 的 kappa RMSE | 0.0267790332 | 0.0005329348 |
| 所有 core 的最大 kappa | 0.3035857587 | 0.0015064603 |

Subhalo 峰位为(0.75,0.075) arcsec，距注入位置0.025 arcsec；同网格离散真峰为0.6314351815。因此原始基线本身已有明显峰压低，不能以新算法没有恢复解析中心尖峰判定优化失败，也不能把与 LM 接近称为质量或质量函数的无偏恢复。[原始验证说明](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/docs/validation/potential-correction-iterative-scientific.md:1)。

## 同一尺度的停止验收

原始指标是

\[
\gamma(x)=\max_i\frac{|\nabla_iF(x)|}{\sqrt{[J(x)^T WJ(x)+Q]_{ii}}},
\qquad \gamma(x)\le a+10^{-8}\gamma(x_0).
\]

Subhalo 的 a=5e-4、初始 gamma=97.6070472082；control 的 a=3e-5、初始 gamma=5.54131377395。约等于5.0097607e-4和3.0055413e-5的阈值是这组 mock 的校准设置。其取值已有严格失败终点和较宽容成功终点的对照记录，不能用任意更宽的新阈值替代原验收。[原始停止控制说明](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/docs/validation/potential-correction-iterative-scientific.md:70)。

新 L-BFGS 的内部 `gtol`、先验白化梯度、完整初始曲率白化梯度都可另外报告；它们不能直接拿上述数值做比较。最终应在**当前终点**重新计算 AD Jacobian 对角 metric 和梯度，并同时保留 SciPy 自身终止原因。原始独立梯度审计也采用16列 panel：[terminal_gradient_audit](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/.scratch/potential-correction/experiments/validate_iterative_scientific.py:19)。

LM 的 `solve_atol=5e-5, solve_rtol=1e-8` 校验其线性增量求解的实际残差；L-BFGS 没有相同内层系统，因此不应伪造对应证书。原始正则化强度为固定值，两个 case 的强度不同；case 间目标函数或 residual 的差不能充当检测显著性。

## 恢复与 control 指标

可以直接复用 [_sample_truth / _recovery_metrics](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/examples/potential_correction/_recovery_diagnostics.py:16)，保留原始单位和域：

- kappa 的 all core=724、illuminated core=526、local core=70；local 半径是 max(2 sqrt(0.04×0.8),2h)=0.3577708764 arcsec，control 使用相同名义位置和圆盘。
- 对照离散真值（解析势在原 U 节点取样后，以原 centered stencil 求导）报告 kappa RMSE、真值 RMS/零势基线、峰值/峰位；解析点采样真值另列。support-only 导数继续为 NaN，不能填零混入统计。
- 报告 raw 源误差、负值比例，raw 势/偏折误差、图像 residual RMS/noise，以及与保存 LM 的源/势/kappa/图像差。仿射对齐如需保留，必须另标 diagnostic，不能替换拟合场。
- control 报告所有/局部 core 的 kappa RMS 与最大正 kappa，并明确该位置是“名义注入位置”。单次 matched control 是伪信号幅度检查，不是已校准假阳性率。
- 若 L-BFGS 未达原停止条件，可以绘图并保留恢复数字，但统一标为“未收敛终点诊断”。`summarize_result` 拒绝失败结果，不应为了复用 exporter 伪造 `success=True`；底层纯 consumer 指标可以直接用于明确标记的诊断对象。[成功边界](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/examples/potential_correction/result_products.py:157)。

## 初始化的含义

与原始 LM 对照的主运行必须复用**解析真源在49²节点的亮度、零势**。这是已知真源初始化的受控算法实验，后续梯度和拟合源仍然从当前像素变量产生，没有冻结解析真源梯度；图像中的源也不是固定的。[原始初始化](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/examples/potential_correction/iterative_scientific.py:203)。

仅由观测图像做 source-only inversion，再启动新算法，是有价值的 observation-only 初始化消融，但改变了非凸问题的起点，需另存结果并明确标记；不能将它与保存的 truth-node LM 终点混称为“同初值换优化器”。即使该消融成功，也不等于宏模型误差、网格选择和真实数据处理已解决。

## 约3,357变量的 AD metric：限制峰值内存

全 Jacobian 的最终 dense 数组为22016×3357 float64，约591 MB。直接全维 `jacfwd` 的射线中间量单数组3357×88064约2.36 GB，四角插值相关中间量可能进一步放大；这是形状算术估计，不是测得峰值内存。

建议把每次 AD 同时传播的方向限制为16（或更小），不构建全单位基矩阵：

1. 对 \(f_w(x)=f(x)/\sigma\)，在初始点调用 `jax.linearize` 获得 \(J_wv\)，用 `jax.vjp` 获得 \(J_w^Tu\)。
2. 16个 one-hot 方向为一组，计算 \(J_w^T(J_we_i)+Qe_i\)，仅输出16×3357的 GN metric 列块；拼接为3357²矩阵（约90 MB），再作固定坐标白化。也可按16列构建 Jacobian、分块转移主机后作 dense 乘法，但应计入其存储/准备成本。
3. 若只需停止指标，保留 \(\|J_we_i\|^2+Q_{ii}\) 即可，不形成完整 metric。末尾不足16个的 panel 可独立编译一次。
4. 初始 metric 仅用于**固定坐标预条件**；L-BFGS 每次目标评估仍通过当前完整非线性 forward，不能把初始 J 固定为物理模型。终点停止指标应重新 linearize。

以上是根据官方 JVP/VJP 定义推导的实现建议：[jax.linearize](https://docs.jax.dev/en/latest/_autosummary/jax.linearize.html)、[jax.jvp](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html)、[jax.vjp](https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html)。完整初始曲率白化仍有平方存储/立方 Cholesky 成本，不能称为已完成可扩展矩阵无关算法。目标、坐标 metric 和数据准备的耗时应分别记账，GPU输出先同步再停止计时。

## 新 runner 与实测状态

已对新 [run_scientific.py](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/run_scientific.py:54) 完成只读科学审查：

- `ad_metric` 按 panel 计算 \(J_w^TJ_w+Q\)，不使用手写解析 Jacobian。`x=x0+L^{-T}z` 白化方向正确，坐标变化不修改物理 prior。
- `inspect_endpoint` 在当前终点重新构建 AD 对角 metric，使用原始 GN 对角尺度做外部验收；解析梯度仅是审计参考，不是 L-BFGS 输入。
- `observation_source` 求解 \((A_w^TA_w+Q_s)s=A_w^T(d/\sigma)\) 的有符号 source-only MAP，条件为零势和固定真宏模型。它没有把数据中的真源亮度当作初始化输入。
- 主程序复用原 JSON、检查保存的图像/mask/noise和网格，分别记录 truth/source-only 初值；未达外部梯度条件的终点保留为诊断对象，无伪造 evidence。

建议补强的 provenance 门已在当前 runner 中加入：核对保存 LM 的 strengths、源 prior 类型和 inversion_nsub。独立 endpoint audit 还检查保存 LM 场在新目标重算与保存目标的差，避免通过 `--configuration` 改变 prior 后仍声称相同问题的 LM 对照。

已只读核对 truth-node 初值的完整原始200²输出：[subhalo diagnostics](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/scientific_full_truth/truth/subhalo/diagnostics.json)、[control diagnostics](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/scientific_full_truth/truth/control/diagnostics.json)。

| 新 AD + L-BFGS，truth-node 初值 | Subhalo | Control |
| --- | ---: | ---: |
| 第三方返回成功 / 外部物理梯度通过 | true / true | true / true |
| 外部 GN 对角缩放梯度 | 3.3554820069e-4 | 4.8221507451e-6 |
| 本 case 原始物理阈值 | 5.0097607047e-4 | 3.0055413138e-5 |
| 新目标 F | 11734.826509001981 | 11404.4436552604 |
| 新 F − 同场重算 LM F | -4.7837584e-6 | -3.9704901e-7 |
| 相对 LM 图像差，RMS/noise | 1.2030501e-5 | 1.8723077e-6 |
| 新 kappa 最大值 | 0.3035761582 | 0.0015066247 |
| 新 all-core kappa RMSE | 0.0267763840 | 0.0005329303 |
| 新非线性 residual RMS/noise | 1.0018374380 | 0.9958284539 |

两者的保存 LM 场在新目标重算与旧保存 F 的差均在约1e-11以内，支持同问题比较。Subhalo 新峰位仍为(0.75,0.075) arcsec，原始恢复与 control 伪信号幅度基本复现。这支持**原始完整 mock、固定先验和已知真源节点初值下的求解器替换可行**；不代表已经校准检测显著性、真实数据表现或速度优劣。

两个 case 的 SciPy 内部终止原因均为函数相对减小，而不是其 `gtol`；外部物理梯度另行检验通过，因此本次接受有独立依据。Subhalo/Control 的 AD 对解析未缩放梯度相对审计误差分别约3.4e-10和2.50e-6；在高精度先验/数据梯度相消时，后者不宜称为未缩放梯度的“全部机器精度一致”。随后独立审计在**同一 GN metric** 下确认差分别为1.07e-13和3.54e-13，解析梯度也分别通过原物理阈值，消除了这一验收疑点。[truth endpoint audit](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/scientific_full_truth/endpoint_audit.json)。

另已只读核对独立的 observation-only source 初值运行：[subhalo diagnostics](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/scientific_full_source_only_v2/source_only/subhalo/diagnostics.json)、[control diagnostics](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/scientific_full_source_only_v2/source_only/control/diagnostics.json)。

| 新 AD + L-BFGS，source-only MAP 初值 | Subhalo | Control |
| --- | ---: | ---: |
| 第三方返回成功 / 外部物理梯度通过 | true / true | true / true |
| 外部 GN 对角缩放梯度 | 3.3348558146e-4 | 1.1430929710e-6 |
| 此初值下同一停止策略阈值 | 5.0016815681e-4 | 3.0000040730e-5 |
| 新目标 F | 11734.826509054532 | 11404.443655259425 |
| 相对保存 LM 图像差，RMS/noise | 1.1950893e-5 | 1.8711437e-6 |
| 新 kappa 最大值 | 0.3035769358 | 0.0015066429 |
| 新 all-core kappa RMSE | 0.0267762303 | 0.0005329379 |
| 初始化线性方程相对残差 | 7.68e-16 | 7.22e-16 |

NPZ 的 CPU 检查确认四个初始联合向量均为3357维 float64、初始势严格为零；truth 初始源与真节点场完全相同，而 source-only 初始源与真节点场的相对 L2 差分别为0.0962868和0.0510371。因此 source-only 运行确实测试了不同的数据驱动起点，未使用真源亮度来启动；固定真宏模型和原源网格仍是此消融的前提。两个运行分别保留 [truth runner snapshot](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/scientific_full_truth/runner_snapshot.py) 与 [source-only runner snapshot](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/scientific_full_source_only_v2/runner_snapshot.py)，其 SHA256 均与各自 environment.json 一致。

[source-only endpoint audit](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/scientific_full_source_only_v2/endpoint_audit.json) 的两个终点均通过，AD/解析梯度在 GN metric 下的差为1.77e-13和3.45e-13；四个终点的重新渲染图像、kappa、同目标 LM 比较和原阈值检查均通过。完整独立检查脚本为 [audit_scientific.py](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/audit_scientific.py:1)。

旧 `scientific_full_source_only/` 因把 MacroMapping 传给需要 SourceMapping 的源渲染入口而在数值优化前退出，不含本表数值；v2 修复为先调用 `source_mapping(...)`，物理模型和目标未变。本报告只使用含完整终点和审计记录的 v2。

可支持的结论是：原始完整 mock 的 subhalo/control，在 truth-node 和 source-only 两种初值下都可由 AD + 固定初始 GN 坐标预条件 + 第三方 L-BFGS 求得与原 LM 实用上相同的重建。这个实验仍使用 dense 初始 metric，并没有证明裸 L-BFGS、全局最优、无初始曲率准备成本、观测宏模型误差鲁棒性或检测统计校准。
