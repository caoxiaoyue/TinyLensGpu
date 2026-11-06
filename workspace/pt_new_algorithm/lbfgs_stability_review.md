# L-BFGS多噪声seed稳定性：设计与只读审查

2026-11-03。仅阅读主源、脚本与保存结果，不运行拟合或GPU任务。用户已明确为seed 0–9，每个subhalo mock运行一次原AD + 固定初始H0白化 + SciPy L-BFGS-B，另有匹配control各一次：共20个L-BFGS终点，而非100次。每个seed/case另运行一次独立GN参考。

## seed改变什么

[生成器](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/examples/potential_correction/mock.py:204) 保持源、宏模型、扰动、PSF、mask与几何固定，只在[噪声抽样](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/examples/potential_correction/mock.py:254) 使用`default_rng(config.seed).standard_normal((200,200)) * noise_sigma`。同seed的subhalo/control共享同一整幅噪声，因此是配对对照；跨seed是10个不同噪声实现，不是10种透镜或源形态。[NumPy 2.4官方说明](https://numpy.org/doc/2.4/reference/random/generator.html) 确认default_rng使用PCG64，并提醒不同版本不保证Generator位流兼容，故版本和实际noise hash均应保存。

初始化为零势下的观测source-only MAP，p0=0，不读取真源节点值；[initializer](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/run_scientific.py:293) 的右端来自观测，先验来自固定源Q。宏模型仍固定真值，源网格和正则化强度仍为原配置。每个独立参考必须使用该seed/case相同的数值初值、观测及prior。

L-BFGS-B本身没有optimizer随机seed；原[PreparedAD](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/benchmark_scientific.py:47) 的起点、H0和更新均由输入确定。更改mock seed改变了目标和source-only初值，不是在随机启动同一个优化问题。JAX的浮点计算和编译重排可能改变细小数值，[官方FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs)；不能仅凭重复运行差异断定具体CUDA算子或插值knot是唯一原因。

## 在新结果前确定的工程政策

主L-BFGS保持原选项：ftol=1e-15、gtol=1e-8、maxcor=30、maxls=80、maxiter=3000、maxfun=120000。SciPy的ftol检查相对目标减小，gtol检查优化坐标的投影梯度，native success并不保证原物理gamma门。[SciPy 1.17官方定义](https://docs.scipy.org/doc/scipy-1.17.0/reference/optimize.minimize-lbfgsb.html)。

独立GN参考使用原物理问题、source-only同初值，cached/explicit/candidate，LSMR 1e-8、ftol/xtol=1e-13、gtol=1e-12。可用参考要求native成功或主动physical-gradient stop，并且数值有限、基本sanity通过，预算停止不可充当参考。原gamma值和严格门另列，不将单分支梯度作为唯一可用性条件。这样的终点是独立、目标值稳定的参考，不能称已证明全局MAP；小步或ftol停止也不证明全局最优。

建议将以下flags分开，最终综合判据可引用它们，但保留每项原始数值：

| 预登记条件 | 解释与适用范围 |
| --- | --- |
| native SciPy success且相关状态/目标/场有限 | API与数值运行状态；不替代其他检查 |
| F_LBFGS − F_ref ≤ 1e-3 | 同seed/case目标的单侧excess，避免把更低F自动判坏；相较F约1e4很小 |
| 图像差RMS/σ ≤ 1e-3 | 有效像素中，差仅为每像素噪声的0.1% |
| 源相对L2差 ≤ 0.005 | 相对参考的0.5%场一致性，不能称真源恢复误差0.5% |
| core κ差RMS ≤ 1e-5 + 0.01 κ_ref_RMS | 参考场幅度的1%加绝对floor；避免control接近零时纯相对误差失真 |
| residual RMS/σ在0.9–1.1 | 宽泛模型sanity；不是严格独立像素χ²检验或优化器收敛证明 |
| subhalo正全局峰距注入≤√2h，h=0.075 arcsec | 网格分辨率尺度的恢复检查；应与“优化/参考一致性”分别报告 |
| control core κ RMS<0.01、paired峰/最大绝对control幅度≥5 | gross-anomaly/幅度sanity，未经检测校准，报告实际值 |

这些数值是本次实验的工程容差。已有seed17先导记录的F差约3e-5、图像差约3e-5噪声，支持1e-3量级一致性门有余量且仍细于噪声；它们没有被证明适用于任意数据、先验或形态。[先导科学记录](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/gauss_newton_performance_review.md)。参考明显更差、两算法找到不同场或参考不可用时，应标为reference mismatch/inconclusive并保留，不能选择性重跑到通过，也不能自动归为L-BFGS失败。

## 科学结论与统计边界

原[科学配置](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/examples/potential_correction/iterative_scientific_configuration.json) 的subhalo/control势强度分别为1.6e7和1e11，因此paired幅度contrast受不同prior影响，不能用来声明检测显著性、假阳性率或对未知数据的检测能力。control κ RMS=0.01约为seed17原0.000533的19倍，只适合gross guard，不能称紧的可忽略误差门。应另外逐seed报告control实际RMS、最大绝对κ、局部场及源/势/图像。

优化一致但参考本身未定位真扰动时，记录“数值一致/恢复检查未过”；参考不可用时记录“未定”，不把它们混为一个优化器失败原因。原seed17重建峰约0.3036、离散真峰约0.6314，已有显著正则化压低；同参考一致不等于无偏峰值或质量恢复。[原验证](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/docs/validation/potential-correction-iterative-scientific.md)。

旧gamma-only失败记录仍按原政策保留：[原timing审查](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/timing_review.md)。此次新的“实际重建一致性”政策、source-only初值和新噪声cohort须明确列出，不能把通过率变化归因于L-BFGS算法已经修复。

10个seed仅探索固定物理模型附近的噪声敏感性。subhalo/control按seed配对，共20次拟合也不是20个独立物理场景；各case报告10次结果和逐seed范围，失败/未定/参考失效分母分开保留。即使全通过，也不据此宣称99%长期可靠性、对初值的广泛鲁棒性或真实数据稳定性。这里只运行一次/seed，也没有直接重测同输入重复运行的数值稳定率。

## Runner只读核对

[run_lbfgs_stability.py](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/run_lbfgs_stability.py:1) 每seed开启新进程，替换MockConfig.seed后重新生成配对观测；同一problem/source-only数值初值分别交给GN与L-BFGS，没有用L-BFGS终点启动参考或反向热启动。主优化仍为原初始AD metric白化和原L-BFGS-B选项，新增callback仅记录历史，不控制迭代。

代码中的差异指标和预登记阈值吻合，所有κ指标及有限性检查限制在724 core，允许support-only导数按原定义为NaN，源仍可为负。峰定位使用√2h，control contrast使用最大绝对κ；独立公式门为AD/解析梯度在同GN metric下差≤1e-8、目标identity误差≤1e-6，区别于原stationarity gamma门。

修订后的归档包括原科学JSON、全部核心Python及装配/恢复代码的目录层次快照/hash与git commit，逐seed后检查未改代码。参考异常独立捕获，仍继续该case的L-BFGS；参考不可用保留inconclusive标记。真实optimizer终点和error records分别计数，避免把异常manifest称为已完成拟合。新目录防覆盖，未安排选择性重试。

已知seed17的`lbfgs_stability_pilot`曾因NumPy布尔JSON序列化中断subhalo记录，属于实现检查；修订使用标量`.item()`序列化。其原始错误manifest保留，不计入预定0–9 cohort，不据此判断求解器科学稳定性。物理/数值阈值不因这个实现修复而改变。

## 正式0–9 cohort的完整核验

[冻结protocol](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_stability_10seeds/protocol.json)、[全部逐seed记录](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_stability_10seeds/summary.json) 与 [final.json](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_stability_10seeds/final.json) 已CPU核对：20个真实L-BFGS终点、0个error records、20个native success、20个工程/恢复sanity综合通过；此次20个也全部通过原legacy gamma门。

归档源码/config全部SHA256与protocol及执行源码一致。逐像素重建`default_rng(seed)`噪声与保存值完全一致，同seed两case共享噪声、10张噪声hash互异；去掉noise后的固定物理图像在浮点容差内一致，源真值/PSF/mask/几何不随seed改变。source-only初值与真源确实不同（relative L2：subhalo 0.09408–0.10129、control 0.04152–0.05827），p0全零，初始化方程残差小于1e-10；每个GN/L-BFGS的保存初值、输入和noise完全相同。未发现借用其他seed数据或真源节点值。

全部endpoint与保存fields一致。已从NPZ重新算active residual、core峰位、κ RMS/最大绝对值、源真值误差、与参考的图像/源/κ差及paired contrast，并重算每个sanity/comparison/overall/inconclusive flag，均与metadata一致。20个L-BFGS均以native相对函数减小停止；未把它们写成gtol停止。AD/解析梯度的GN metric差最大3.87e-13，目标公式误差最大3.64e-12。

| 正式10 seed，每case一次 | Subhalo | Control |
| --- | ---: | ---: |
| native / 新综合门 / legacy gamma通过 | 10 / 10 / 10 | 10 / 10 / 10 |
| fresh gamma范围 | 5.64e-5–2.93e-4 | 2.72e-8–7.18e-6 |
| F_LBFGS − F_GN范围 | -4.53e-5至+2.88e-6 | -9.92e-7至-2.19e-9 |
| 最大图像差RMS/σ | 3.74e-5 | 1.99e-6 |
| 最大源相对L2差 | 7.32e-5 | 2.81e-6 |
| 最大core κ差RMS | 3.29e-5 | 1.90e-7 |
| active residual RMS/σ范围 | 0.99597–1.00376 | 0.98889–0.99637 |
| source relative L2真值误差范围 | 0.06084–0.07355 | 0.04165–0.05725 |

20个参考均按预登记可用，但停止类型不同：18个在physical-gradient callback通过后停止（native status=-2/success=false）；seed3、5的subhalo参考由native ftol停止（status=2/success=true），其gamma分别1.07915e-3、5.89392e-4，高于各自约5.0017e-4的旧门。它们不能写成旧梯度验收通过；独立公式一致性及新参考可用性门实际通过，且L-BFGS分别比它们低F约1.84e-5、6.06e-6、场差很小。无参考被丢弃、替换或选择性重跑。

Subhalo的10个正全局峰都在(0.75,0.075) arcsec，距注入位置0.025 arcsec；峰范围0.292982–0.312172，均值约0.306723，seed间样本标准差约0.005398。它们仍远低于离散真峰0.631435，prior压低未消失。Control实际κ RMS为0.0005434–0.0015116，最大绝对κ最高约0.005614；paired幅度contrast为54.4–195.3。这些是固定不同prior下的实际场诊断，不是检测显著性。

本次支持原方法在这10个新增噪声实现、数据驱动源初值和原固定物理问题下稳定求得与独立参考一致的重建。它没有重测同输入多次运行，不能消除原truth-init warm实验已有的数值重复性问题；也不代表跨形态、宏模型误差、正则化/evidence外层优化或真实数据的长期保证。本批每seed新进程并含新编译，耗时不与此前warm benchmark直接相比。

最终 [结果说明](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_stability_results.md) 与 [assessment.json](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_stability_10seeds/assessment.json) 已复核；全部case统计的min/max/mean、参考停止计数和总计数与原始记录一致，未发现材料不符。
