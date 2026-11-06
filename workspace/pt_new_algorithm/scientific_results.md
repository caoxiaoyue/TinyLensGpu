# 原始完整 Mock：自动微分 + 固定白化 + L-BFGS

2026-11-03。**完整 200×200 引力透镜扰动恢复实验通过。** 新方法在原始真值源初值、以及仅由观测反演的源初值下，均通过原实验的物理梯度门限，并恢复到与现有 LM 几乎一致的图像、源和扰动场。四个终点已独立复核。

这支持更换内层求解器的可行性。验证的具体方法是“JAX 自动微分 + 一次初始 Gauss–Newton 白化 + 第三方 L-BFGS”，不是没有任何曲率准备的裸 L-BFGS；本轮不作系统整合。

## 实验保持了什么

直接使用 [原始科学配置](../../examples/potential_correction/iterative_scientific_configuration.json)，没有缩小图像或修改物理先验：

| 项目 | 设置 |
| --- | --- |
| 图像 / 有效数据 | 200×200 / 22,016 像素 |
| 像素尺度、生成/反演子像素积分 | 0.012 arcsec；2 / 2 |
| 宏模型 | 固定真值 SIS，Einstein radius 0.8 arcsec；shear (0.05, 0.02) |
| 注入扰动 | 解析 Pseudo-Jaffe；b_sub=0.04；位置 (0.75, 0.1) arcsec |
| 控制图像 | 零扰动，与注入图像使用同一噪声实现 |
| PSF / 噪声 | 27×27 Gaussian，sigma=0.05 arcsec；噪声 sigma=0.01，seed=17 |
| 源网格 | 49×49，bbox=[−0.4, 0.4]²，共 2,401 变量 |
| 势网格 | 间距 0.075 arcsec；956 变量，其中 724 个导数有效 core 节点 |
| 总变量 / 查询子像素 | 3,357 / 88,064 |
| 源正则化 | 原 second-order，lambda_source=0.0006 |
| 势正则化 | 原完整四阶差分；subhalo/control 的 lambda_potential=1.6e7 / 1e11 |
| 势幅度先验 | lambda_amplitude=10000，作用于全部势节点 |
| 精度 / 设备 | float64，JAX 0.7.1，SciPy 1.17.1，CUDA RTX 5080 |

图像由原 mock 生成器的连续解析源和解析扰动偏折独立生成，包含 PSF、噪声和子像素积分；随后用像素源和像素势反演。这是完整 scientific mock，不是同一节点模型生成再反演的 nodal 测试。

比较基线为之前已验证并保存的原始 dense LM 结果：
[subhalo](../../.scratch/potential-correction/output/iterative-scientific/validated-reference/subhalo/dense/result.json)、
[control](../../.scratch/potential-correction/output/iterative-scientific/validated-reference/control/dense/result.json)。
不是从文档抄取几个标量：本次实际加载其 NPZ 场，核对图像、mask、噪声、网格和配置，并在当前目标函数中重新计算。保存的 LM 场重算目标与原记录差约 1e-12。该基线的历史收敛说明见 [原验证记录](../../docs/validation/potential-correction-iterative-scientific.md)。

## 新方法如何运行

每次目标评估都重新执行完整非线性 forward，联合优化有符号源亮度和势修正。目标仍为 0.5×(chi² + source penalty + potential penalty)，没有重缩放物理场、删减低阶势模式或改变正则化。

在每组初值处，用 JAX 的 JVP/VJP 计算

\[
H_0=J_{w,0}^{T}J_{w,0}+Q,\qquad f_w=f/\sigma,
\quad x=x_0+L^{-T}z,\quad LL^T=H_0.
\]

按 16 列分块直接构造 H0，避免同时存储全部图像 Jacobian 的巨大中间量。Cholesky 不加隐藏 jitter；白化在整个 L-BFGS 过程中保持不变。目标梯度由 JAX 反向微分计算，优化使用无 bounds 的 SciPy L-BFGS-B。参数为 maxiter=3000、maxcor=30、maxls=80、gtol=1e-8、ftol=1e-15。

完整 H0 仍是 3357×3357 dense 矩阵，单矩阵约 90 MB。分块控制了准备过程的显存，但没有消除完整矩阵及分解成本，不能称为已完成大规模 operator 实现。

## 两种初值都测了

**原实验初值（truth）**：源亮度由模拟真值在源节点上采样，初始势修正为零。它与保存的 LM 使用相同起点，直接检验更换求解器。

**观测源初值（source_only）**：保持势修正为零，在给定宏模型下，仅利用观测图像、噪声和原源正则化，解一次有符号线性源反演。该步骤及后续白化都不读取真实源亮度或真实扰动。两幅图像的初始源与真值相对 L2 距离分别为 9.63% / 5.10%，并不等于真值；源方程相对残差为 7.68e-16 / 7.22e-16。

source_only 是额外的初始化实验。其与保存 LM 的比较是最终场参照，不是相同初值的迭代次数对比。**宏模型仍固定为真值**，本轮未测试宏模型误差或从原始观测开始的宏模型寻优。

## 收敛不是只看 success

沿用原实验的当前 GN 对角尺度梯度：

\[
\gamma=\max_i\frac{|\partial F/\partial x_i|}
 {\sqrt{[J_w^TJ_w+Q]_{ii}}}.
\]

每个终点都重新通过 AD 计算当前对角；没有拿初始白化坐标中的梯度替代它。门限仍是原策略的 gradient_atol + 1e-8×初始 gamma，subhalo/control 的 atol 分别为 5e-4 / 3e-5。

| 初始源 | 案例 | L-BFGS 步数 | 最终 F | 最终 gamma | 实际门限 | 结果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 真值节点采样 | subhalo | 246 | 11734.826509002 | 3.3555e-4 | 5.0098e-4 | 通过 |
| 真值节点采样 | control | 13 | 11404.443655260 | 4.8222e-6 | 3.0055e-5 | 通过 |
| 观测反演 | subhalo | 305 | 11734.826509055 | 3.3349e-4 | 5.0017e-4 | 通过 |
| 观测反演 | control | 16 | 11404.443655259 | 1.1431e-6 | 3.0000e-5 | 通过 |

四个 SciPy 终点均因目标相对变化小而返回成功；上表是另行验证后的接受结果。再次用现有解析梯度独立核对，在相同 GN 尺度下 AD 与解析梯度最大差为 1.07e-13 到 3.54e-13，均远小于门限。

未缩放的 control 梯度差不能概括为机器精度：其原始差指标约 2.5e-6 / 3.0e-6，受大正则化项的浮点抵消影响。实际采用的物理尺度核对给出了上述约 1e-13 差异。

## 扰动究竟恢复得怎样

subhalo 的真实离散 convergence 峰为 0.631435，比较使用原来的同网格中心差分真值，不混用奇点附近的解析点值。

| subhalo 指标 | 原 LM | 新方法，真值源初值 | 新方法，观测源初值 |
| --- | ---: | ---: | ---: |
| 恢复峰值 | 0.30358576 | 0.30357616 | 0.30357694 |
| 峰值位置 / arcsec | (0.75, 0.075) | (0.75, 0.075) | (0.75, 0.075) |
| 距注入位置 / arcsec | 0.025 | 0.025 | 0.025 |
| 源相对 L2 误差 | 约 0.07029 | 0.0702882 | 0.0702874 |
| 局部 convergence RMSE | 约 0.058278 | 0.0582776 | 0.0582774 |
| 图像残差 RMS / noise | 约 1.001837 | 1.00183744 | 1.00183746 |
| 与原 LM 图像差 RMS / noise | — | 1.2031e-5 | 1.1951e-5 |

原来约 **52% 的峰值压低仍在**。更换优化器恢复了相同正则化问题的解，没有改善这项物理恢复偏差。

控制图像的 all-core convergence RMS 分别为 **0.0005329303 / 0.0005329303**，与原 LM 的 **0.0005329348** 一致；最大正起伏约 **0.0015066**，原 LM 为 **0.00150646**。因此，新求解器没有在这组对照中引入额外可见的扰动恢复差异。subhalo/control 使用不同固定平滑强度，这不是检测显著性或假阳性率校准。

![原初值下的扰动恢复对照](scientific_figures/truth_subhalo_perturbation.png)

![观测源初值下的源恢复](scientific_figures/source_only_subhalo_source.png)

![观测源初值下的图像及残差](scientific_figures/source_only_subhalo_images.png)

图像差面板有独立且明确标注的色标，其量级约 1e-5 noise；不能把颜色深浅与 ±4 noise 的残差面板直接比较。全部图保留原始有符号场，导数图仅显示有效 core，不进行 affine alignment 或负值裁剪。

## 代码、结果和复现

- 完整实验：[run_scientific.py](run_scientific.py)。
- 保存终点独立复核：[audit_scientific.py](audit_scientific.py)。
- 从保存数据绘图：[plot_scientific.py](plot_scientific.py)；不重新拟合。
- 原初值结果：[summary](scientific_full_truth/summary.json)、[endpoint audit](scientific_full_truth/endpoint_audit.json)。
- 观测源初值结果：[summary](scientific_full_source_only_v2/summary.json)、[endpoint audit](scientific_full_source_only_v2/endpoint_audit.json)。
- 数学、配置与结果的独立只读核对：[scientific_review.md](scientific_review.md)。

每组保留配置、环境、确切 runner 快照及 SHA256、L-BFGS 历史、完整源/势/图像/真值、检查点及状态。矩阵缓存按配置、起点、噪声和物理强度区分。所有数值输出均继续由 gitignore 忽略，脚本与说明可追踪。

在仓库根目录、tinylens_gpu 环境中，一次重现两种初值和两个案例：

~~~bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python -u workspace/pt_new_algorithm/run_scientific.py \
  --output workspace/pt_new_algorithm/scientific_recheck
python -u workspace/pt_new_algorithm/audit_scientific.py \
  --runs workspace/pt_new_algorithm/scientific_recheck
python workspace/pt_new_algorithm/plot_scientific.py \
  --runs workspace/pt_new_algorithm/scientific_recheck \
  --output workspace/pt_new_algorithm/scientific_recheck/figures
~~~

要求新输出目录，以保留既有数值证据；缓存可复用，需测准备成本时使用新的 --metric-cache。

验证结果：

- 四个完整终点的独立审计全部通过，包含原梯度门限、解析/AD 一致性、保存场重渲染、原 LM 同目标复算、扰动位置及控制场核对。
- 在 examples/tests/ 单独运行 test_potential_correction_iterative_scientific.py 和 test_potential_correction_scientific.py：**9 passed**。
- 已目视检查扰动恢复、观测源初值、图像和残差图。
- 没有计算新 evidence、调整超参数、估计扰动质量区间或检测显著性。没有据保存的 LM 历史运行与本次 L-BFGS 组件计时计算加速比。

当前结论仅覆盖这套固定宏模型、固定网格和先验的 Mock。首轮 32×32 粗网格实验的停滞记录仍然保留；本次成功不等于任意初值、任意离散化和真实数据都已验证。

后续 [速度补测](timing_results.md) 显示成功运行有明显耗时优势，但在复用编译函数的重复测试中，也发现了 native success 后未通过原梯度门限的终点。此前四个成功恢复记录保持不变；稳定性结论应同时参照这些新增失败，不能仅凭本页的四个成功端点推导普遍保证。
