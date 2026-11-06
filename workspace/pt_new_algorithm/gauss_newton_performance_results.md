# 高斯–牛顿性能拆解与优化实验

2026-11-03。原200×200 Mock、RTX5080、JAX0.7.1/SciPy1.17.1、float64、16 BLAS线程。源和势网格、PSF、mask、同一噪声、宏模型、初值与全部物理先验保持原配置。没有修改核心库或整合接口。

问题是为什么SciPy TRF/LSMR高斯–牛顿比之前的L-BFGS慢，以及哪些部分可以优化。先重放已保存计时，再串行运行每次只改一项的对照；接受仍要求原当前GN尺度梯度、额外独立AD/解析证书及恢复场核验。

**结论：主要差距来自大量内层导数动作、重复的精确GN尺度计算，以及每动作反复应用固定白化。** 可选组合已把subhalo warm中位从8.96秒降至2.71秒，control从2.24秒降至1.16秒；本配置约快3.30倍/1.94倍，仍慢于历史L-BFGS通过样本。原始目标、先验与最终梯度门未改变。三轮共78个同一Mock的配置/重复终点全部通过独立验收；这些不是78个独立数据集，不能推导普遍成功率。

## 慢在什么地方

第一轮新鲜baseline的两次warm平均，总耗时约8.88秒（有扰动）、2.23秒（control）。与历史8.79–9.07秒、2.22–2.24秒一致。下列各项互斥，子项不重复累加：

| 完整fit中的项目 | subhalo | control |
| --- | ---: | ---: |
| 导数动作Jv、Jᵀu及批量动作 | 约4.68秒 | 约0.24秒 |
| 逐轮收敛检查 | 约3.34秒 | 约1.28秒 |
| 初始metric、分解及准备 | 约0.53秒 | 约0.54秒 |
| 最终证书和原始场产品 | 约0.15秒 | 约0.15秒 |
| 残差、其余SciPy工作及边界开销 | 约0.18秒 | 约0.02秒 |

逐轮检查内部，subhalo的物理目标/梯度只占约0.029秒；重新计算当前GN对角占约3.31秒。control相应为0.011秒和1.27秒。大头是重算尺度，不是求一个梯度。

本轮subhalo虽然只走23–24次外层迭代，却在线性求解接口调用约3200次Jv/Jᵀu；逐轮GN尺度还另做分块、多方向的微分。每轮LSMR仍需多次线性迭代。历史L-BFGS通过样本为192/249次目标加完整梯度评估，优化本体约0.41/0.47秒，且只在终点计算一次完整GN尺度。因此外层步数少并不代表工作量小。

SciPy除了已单列的导数动作和检查，其余外层/内层控制与向量运算约0.13秒/subhalo、0.01秒/control。单纯减少Python日志或换一个包装函数，不能解决数秒的主要差距。

## 单项对照：先验证三个假设

每方案独立进程、每case首调用加2次warm。warm保留编译函数，但每次重算数值H0与分解，不复用数值白化结果；各方案串行，无重叠GPU工作。第一轮共36个终点，**全部通过原物理门及独立AD/解析检查**。

| 唯一改变的项目 | subhalo warm中位/秒 | control warm中位/秒 |
| --- | ---: | ---: |
| baseline：每轮完整检查、direct动作、内层1e-10 | 8.881 | 2.231 |
| 候选点才做完整检查 | 6.507 | 1.249 |
| 仅将LSMR容差改1e-8 | 8.008 | 2.189 |
| 仅将LSMR容差改1e-6 | 6.909 | 2.149 |
| 仅缓存当前点的线性化 | 7.886 | 2.223 |
| 候选检查 + 缓存 + 内层1e-8 | 4.760 | 1.195 |

候选策略先计算当前物理梯度在初始metric对角下的尺度；它大于原阈值10倍时暂不重算当前对角，接近时才启动完整检查。subhalo完整callback检查由23–24次变成6次，control由9次变成2次。该筛查只是调度启发式，初始尺度没有严格界，可能错过最早合格点；所有接受点仍须通过当前真实对角和独立终点证书，不会用近似尺度直接接受。

LSMR容差是每轮线性子问题的停止精度，改变它不改物理Q或非线性F。1e-10在本例偏紧；1e-6使subhalo的Jv/Jᵀu从约3200次减为约1800次。这里通过不代表任意数据都适用1e-6，原物理终点门始终保留。[LSMR官方容差定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html)。

缓存方案每到一个新的jacobian点，通过jitted factory调用JAX linearize，保存这个点的线性算子；同一LSMR求解中复用它及其转置。跨点重新构造，不能冻结首轮Jacobian。各cached factory/apply/transpose/batch kernel的编译缓存大小均为1，没有逐轮重编译。增加的linearization构造时间约0.027秒/subhalo，已经计入。[JAX linearize](https://docs.jax.dev/en/latest/_autosummary/jax.linearize.html)。

## 数据传输不是动作成本的大头

另跑同步分段诊断，每case首调用加一次warm，4个终点均通过。在每次动作内分别同步输入、设备调用和输出：

| warm动作内子计时 | subhalo | control |
| --- | ---: | ---: |
| 主机到设备 | 0.228秒 | 0.0115秒 |
| 设备调用wall time | 4.417秒 | 0.225秒 |
| 设备到主机 | 0.148秒 | 0.0075秒 |
| 完整动作总计 | 4.795秒 | 0.244秒 |

输入输出约占动作8%。设备调用段包含dispatch和等待，不能称为纯CUDA kernel时间；这些诊断也没有覆盖callback/分解/残差的全部传输。三段是动作总计的子项，不能再与动作总计相加。同步分段会扰动执行，因此该批只作定位，不代替主速度表。

## 更具体的瓶颈：反复应用固定白化

当前白化为x=x0+L0^{-T}z，L0在一次fit中固定。原实现每次Jv/Jᵀu仍要解一次三角方程；缓存非线性线性化并不会消除这部分工作。

在保存的subhalo终点，以固定设备输入、预热、40调用×3组、随机组件顺序测得中位数：

| 单次同步设备调用 | wall time |
| --- | ---: |
| 白化三角求解 | 0.424毫秒 |
| 白化转置三角求解 | 0.635毫秒 |
| 显式T矩阵乘法 | 0.133毫秒 |
| 物理图像JVP / VJP | 0.120 / 0.231毫秒 |
| 完整direct JVP / VJP | 1.315 / 1.417毫秒 |
| 完整cached JVP / VJP | 0.829 / 1.206毫秒 |
| cached + 显式T的JVP / VJP | 0.350 / 0.474毫秒 |

组件存在融合和不同的数据流，不能简单相加。这是该终点/向量/硬件的微基准，不是端到端加速比，也不是纯kernel时间。

显式T=L0^{-T}由一次solve_triangular(L0.T,I)得到；后续使用T@z及T.T@u，仍然是同一固定坐标变换。微基准首次构造T用时0.187秒，额外约90.16MB；该时间含首次形状/运行时成本，不能当作warm常数。规范化逆残差4.09e-15、随机向量相对变换误差1.96e-15；与原direct JVP/VJP的比较通过。完整拟合仍需计入每次构造与独立终点验收。

显式变换又做了同批端到端对照，每case仍为首调用加2次warm，**24/24终点通过原独立物理门**：

| 第二轮方案 | subhalo warm中位/秒 | control warm中位/秒 |
| --- | ---: | ---: |
| 本轮baseline | 8.959 | 2.240 |
| 仅显式T，其他为baseline | 6.415 | 2.182 |
| 候选检查 + cached + 内层1e-6，仍用三角求解 | 3.793 | 1.168 |
| 上述组合再加显式T（fast） | **2.711** | **1.157** |

每fit均重新构造T，warm约0.0665秒，已包括在factor/setup/total，不能再额外加一次。全部初始方向变换核对、原目标身份、cached动作与direct动作核对均通过；独立AD/解析梯度在相同GN尺度下差约4e-13量级。

在subhalo中，组合方案动作约2.064秒降至0.957秒，可以摊薄T的准备成本；control动作只节省约0.056秒，接近准备成本，因此1.168→1.157秒的约1%差异不足以证明显式T对小调用量case额外提速。每份T本身约90.16MB，不是完整进程显存峰值。第一、第二轮各有自己的代码快照，未覆盖任何原始记录。

## 最后一项：把精确对角检查放入编译循环

原panels后端每次精确检查从Python发出约210批、每批16方向的JVP，再把小结果取回。新增compiled后端，在当前physical x只linearize一次，通过lax.map逐批生成16个onehot方向、计算并立即归约其图像列范数，最后加原Qdiag：

\[
\mathrm{diag}(H_{GN})_i=\|J_w(x)e_i\|^2+Q_{ii}.
\]

这是同一个精确对角，不是随机估计、旧点对角或放宽门限；中间只保留一批，不构造完整Jacobian。初始值与原H0对角核对，终点仍用旧AD/解析独立证书验收。该改动同时减少Python边界调用及重复的同点线性化；不能只归因于数据复制。

第三轮同批记录，每方案仍首调用加2次warm，18/18终点均通过：

| 第三轮方案 | subhalo warm中位/秒 | control warm中位/秒 |
| --- | ---: | ---: |
| fast：候选检查、panels对角 | 2.742 | 1.158 |
| compiled对角，每轮完整检查 | 4.938 | 1.982 |
| compiled对角，候选点完整检查 | 2.638 | 1.191 |

compiled单次精确对角约0.113秒，panels约0.143秒，仅快约21%；额外初始一致性audit又将setup从约0.61提高到0.72秒。每轮检查实际用了25–27次完整对角（约2.94秒），相对候选策略6次（约0.68秒）更慢。这里同时存在数值轨迹/步数的小幅变化，不能假定所有动作数恒定。

同为候选策略时，subhalo快约3.8%、control慢约2.8%；仅2个warm样本，不能认定是明确的整体优化。保留该实验及负/混合结果，**推荐的组合仍使用panels对角，不将compiled设为默认**。这也反驳了“把Python循环简单搬进编译计算，就能大幅消除精确对角成本”的初始猜测。

compiled初始对角与原H0最大相对差1.37e-15，所有kernel缓存保持1，物理/解析梯度差约4.4e-13量级。此处收益有限来自实际工作量和准备成本，不是靠改变公式获得更松的证书。

## 优化后剩余的差距

以第二轮fast的两次warm平均为例，互斥分解为：

| 项目 | subhalo | control |
| --- | ---: | ---: |
| 初始准备（含每次重建T） | 0.601秒 | 0.605秒 |
| Jv/Jᵀu/批量动作 | 0.957秒 | 0.079秒 |
| 每点线性化准备 | 0.019秒 | 0.008秒 |
| callback筛查及精确验收 | 0.881秒 | 0.292秒 |
| 残差及其他SciPy工作 | 0.102秒 | 0.017秒 |
| 终点证书/产品与边界开销 | 0.152秒 | 0.157秒 |
| **完整fit** | **2.711秒** | **1.157秒** |

历史L-BFGS通过样本为1.07–1.12秒、0.70–0.72秒；其有扰动warm还保留了1次物理门失败。原GN约8倍的有扰动差距已缩至约2.5倍，control约缩至1.6倍；这是不同批次的具体实现成本比较，不是包含失败重试的通用吞吐率。

剩余空间主要在减少内层动作/设备调用，以及验收调度；把整个内层迭代留在GPU可能降低调用/同步成本，但本次没有实现该后端。初始H0及分解也是双方共有的固定成本，跨fit复用数值矩阵没有纳入本次公平计时。继续压缩检查需要保留真实终点门；候选策略可能延后发现最早合格点，不能将其描述为与每轮检查完全相同的停止路径。

## 观测初值与失败退出检查

推荐fast组合另在两例source-only初值上验证，source_truth_used=False、p0=0，宏模型仍固定真值：

| case | F | 当前梯度 / 原门限 | 结果 |
| --- | ---: | --- | --- |
| subhalo | 11734.8265412 | 4.99640e-4 / 5.00168e-4 | 通过 |
| control | 11404.4436554 | 9.65335e-6 / 3.00000e-5 | 通过 |

Subhalo峰约0.303554，仍在(0.75,0.075) arcsec；control kappa RMS约0.000532962，与原解相符。它们支持观测源初值下的可行性，不作为新的warm速度样本。

数值算子另用[audit_gauss_newton_operators.py](audit_gauss_newton_operators.py)在CPU上检查两种toy、两种白化和三个变化点，覆盖目标身份、梯度链式法则、cached/direct动作、adjoint、Jᵀr=gradient及compiled/panels精确对角，12个跨点配置全部通过，见[完整输出](gn_perf_operator_audit.log)。该独立脚本首次导入曾读到环境中的另一个checkout，已修复仓库路径初始化后重跑；主计时脚本始终显式设置当前仓库路径，不受该检查脚本问题影响。

[评估预算退出](gn_perf_eval_budget/summary.json)为status=0，[wall预算退出](gn_perf_wall_budget/summary.json)为status=-2、callback_wall_budget，两例均accepted=False、独立物理门未过、实际终点有限且与history一致，并按设计返回退出码1。它们单独保留诊断，不能计入成功耗时。

## 范围、实现与复现

新增[串行对照脚本](profile_gauss_newton.py)和[组件微基准](microbench_gauss_newton.py)，在[原GN脚本](run_gauss_newton.py)增加可选参数。默认仍保留原every/direct/1e-10/solve配置，历史结论及原始快照不被新方案替换。

没有新增物理ridge、质量片重标定、非负约束，也没有放宽最终物理梯度门。所测宏模型固定真值；主计时用相同truth-node源初值。全过程仍有dense初始H0及分解，不能称为线性存储的全矩阵自由算法。

时间从已准备problem开始，包含初始metric、全部分解/变换准备、初始目标/导数核对、全部TRF/LSMR动作、callback检查和终点场产品。排除Mock/几何生成、计时后的残差身份复核、最终文件写入、绘图及返回后的额外独立解析审计。各方案保存代码快照、环境、输入hash和完整逐轮记录，失败样本也保留。

~~~bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python -u workspace/pt_new_algorithm/profile_gauss_newton.py \
  --output workspace/pt_new_algorithm/gn_performance_recheck

python -u workspace/pt_new_algorithm/profile_gauss_newton.py \
  --variants baseline explicit combined6 fast \
  --output workspace/pt_new_algorithm/gn_whitening_recheck

# 推荐的可选组合；每次重新准备数值矩阵，最终门保持原值
python -u workspace/pt_new_algorithm/run_gauss_newton.py \
  --gate-policy candidate --linearization cached --lsmr-tol 1e-6 \
  --whitening explicit --repeats 2 \
  --output workspace/pt_new_algorithm/gn_fast_recheck

# 小型、确定性的跨点算子检查，不是性能测试
JAX_PLATFORMS=cpu python workspace/pt_new_algorithm/audit_gauss_newton_operators.py
~~~

证据：[第一轮summary](gn_performance_study/summary.json)、[同步分段](gn_performance_bridge/summary.json)、[设备组件](gn_performance_micro/results.json)、[显式变换同批对照](gn_performance_whitening/summary.json)、[对角计算对照](gn_performance_diagonal/summary.json)、[观测初值](gn_perf_source_only/summary.json)、[独立主源与实现核对](gauss_newton_performance_review.md)。
