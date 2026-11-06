# Gauss–Newton性能分解与改进审查

2026-11-03。仅审查主源、代码和保存数值；本审查不运行拟合或GPU工作。范围为原200²、3357维、固定原强度与truth-node初值的同一MAP问题。

## 保存基线能解释的成本

CPU重算 [GN完整基线summary](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/gauss_newton_full_truth/summary.json) 每case两次warm的均值：

| 互斥fit内成本，秒 | Subhalo | Control |
| --- | ---: | ---: |
| 初始metric、factor及初始audit | 0.53318 | 0.53829 |
| 当前Jacobian动作 | 4.73787 | 0.23684 |
| callback物理梯度/当前GN对角检查 | 3.35406 | 1.27832 |
| 残差桥接 | 0.02859 | 0.01005 |
| 最终证书与products | 0.14903 | 0.15239 |
| 余下控制及计时开销 | 0.12668 | 0.01088 |
| 总计 | 8.92940 | 2.22676 |

Subhalo的Jacobian动作与callback分别占53.1%和37.6%，每次平均3242个matvec/rmatvec/matmat调用；control只有155个动作，callback却占57.4%。这些动作的墙钟同时包含设备计算、数据传输、同步、dispatch和Python封装，不能把4.738秒全部称为GPU计算。原实现每个动作都从当前点调用完整JVP/VJP，而callback每轮重新算3357项当前GN对角：[原归档runner](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/gauss_newton_full_truth/runner_snapshot.py:148)。L-BFGS不要求反复求内层GN线性最小二乘，因此外层迭代少并不保证GN总动作少；这里的瓶颈有明确实现成本，不能泛化为所有GN均更慢。

作为工作量参照，先前 [L-BFGS subhalo记录](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/timing_full/lbfgs_subhalo.json) 的两个成功warm样本是192/249次value-and-gradient调用，尽管有157/193次外迭代；GN基线只有23–24轮外迭代，却需数千次Jacobian动作。两种调用并非等价成本单位，但能解释为何不能只比较外迭代数。此前另一条L-BFGS warm样本失败也应继续保留，不能用成功样本速度忽略验收比例。

## 一次线性化、多次动作的数学与缓存条件

令原augmented残差为

\[
r(z)=\begin{bmatrix}(f(q_0+L_0^{-T}z)-d)/\sigma\\L_Q^T(q_0+L_0^{-T}z)\end{bmatrix},\quad F=\tfrac12\|r\|^2.
\]

对每个新的Jacobian点\(z_k\)，`r_k, lin_k = jax.linearize(residual, z_k)`给出精确AD线性映射\(lin_k(v)=J_kv\)。它缓存该点需要的中间值，反复动作不再重复线性化，代价是保存更多设备中间数组。[JAX linearize官方文档](https://docs.jax.dev/en/latest/_autosummary/jax.linearize.html)。`jax.linear_transpose(lin_k, tangent_spec)(u)[0]`得到\(J_k^Tu\)，只适用于这里已线性化的函数，不能直接转置非线性residual。[官方linear_transpose](https://docs.jax.dev/en/latest/_autosummary/jax.linear_transpose.html)。

已安装JAX0.7.1将返回的`lin_k`显式包装为pytree `Partial`：[api.py](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/jax/_src/api.py:1988)。其bound arrays是动态pytree children，函数是静态结构：[tree_util.py](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/jax/_src/tree_util.py:600)。因此可以用稳定的jitted factory返回lin，再通过稳定的`jax.jit(lambda lin,v:lin(v))`和jitted transpose应用；官方也支持把Partial传给JIT。[Partial文档](https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.Partial.html)。相同结构/shape/dtype允许复用编译，改变点的数组值本身不要求重新编译；每轮临时`jax.jit(lin_k)`、把整个lin设成static、或改变其静态函数身份会破坏这一条件。[JIT缓存说明](https://docs.jax.dev/en/latest/jit-compilation.html)。

每个LinearOperator须拥有自己当轮的lin，接受新点后必须重新创建，不能沿用旧点导数。bilinear插值的分支导数应与直接JVP/VJP一致，不能为了复用而冻结旧cell。需要检查方向导数、\(u^TJv=v^TJ^Tu\)、与direct动作一致，以及实际编译缓存是否稳定。若复用\(J_k^Tr_k\)求gradient，它是白化坐标梯度\(g_z\)；物理梯度应为\(g_q=L_0g_z\)，不是\(L_0^Tg_z\)或再做三角求解。独立物理证书保留，尤其接近驻点时不能只用相消后的相对梯度误差解释精度。

## 降低callback与内层精度的边界

候选筛选可先算\(\gamma_0=\max_i|g_i(q)|/\sqrt{(H_0)_{ii}}\)，只在其接近阈值（例如10倍阈值内）时计算昂贵的当前\(\gamma(q)=\max_i|g_i|/\sqrt{H(q)_{ii}}\)。**只有后者及最终独立AD/解析门通过才能接受**。没有当前/初始对角比的界时，这只是启发式检查调度：它能推迟或错过最早合格点，不能称为与每轮精确检查完全相同的停止轨迹。墙钟/评价预算检查仍需正常执行；native gtol提前退出也必须通过最后的原物理门。

把LSMR的`atol/btol`从1e-10放宽到1e-6可减少内层动作，但改变了GN方向的求解精度和后续轨迹，不是单纯缓存优化。它们控制线性子问题的backward-error条件，不是外层物理驻点精度：[SciPy LSMR文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html)。TRF随后用真实目标下降调整信赖域，算法阻尼仍与物理prior不同：[已安装trf.py](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/scipy/optimize/_lsq/trf.py:487)。应先分别测试cached、candidate、looser-LSMR，再测试组合；所有失败终点和相同物理门应保留。不能用低动作数或小F差替代收敛验收。

## 计时解释

独立benchmark应保持float64、相同数据/初值/强度、串行执行，分别报告冷编译和复用kernel后的fit；初始metric/factors每fit重建，最终物理门与products计入。昂贵独立审计是否在fit计时外须明确，不能从某一方案选择性扣除。

桥接profile可以顺序分成：①输入放设备并同步；②jitted动作并`block_until_ready`；③结果变NumPy并计时。第二项是**设备调用墙钟（含dispatch/wait）**，不是纯GPU kernel时间；真实kernel/拷贝/调度归属需要设备trace。额外同步也会扰动真实运行，应与关闭profile的总fit分开报告。[JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html)、[async dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)、[profiling](https://docs.jax.dev/en/latest/profiling.html)。

factory的Partial也必须同步全部array leaves。现有`benchmark_scientific.sync`只遍历dict/list/tuple/dataclass，会忽略Partial；直接用`jax.block_until_ready(lin)`或显式遍历pytree leaves。[已安装pytree同步实现](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/jax/_src/api.py:3021)。每个动作已有NumPy转换时其总桥接时间原本已同步；拆分profile不能把同一等待重复计入总耗时。

## 新探针代码的只读审查

[新增可选路径](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/run_gauss_newton.py:80) 保留默认every/direct/1e-10；cached factory在每次jac调用创建当点Partial，operator闭包各自保存`arguments=(linear,)`，使用pytree同步，transpose只转置linear，batched matmat方向正确。初始cached/direct JVP/VJP比较和原目标/方向/adjoint audit保留；最终current-diag证书及独立AD/解析证书均未省略。未发现改变物理目标或误用旧点导数的问题。

candidate没有把screen当作验收分数：只有计算了当前对角并通过才主动停止；skip/full-check次数与每轮记录可追查，预算检查仍执行。更宽LSMR容差按独立variant试验，`native success/status`与外部物理门仍分别保存。cache size仅作诊断，不参与算法决策。

[profile_gauss_newton.py](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/profile_gauss_newton.py:38) 的decomposition将setup、linearization、actions、callback、residual、余下solver控制、endpoint和outside-accounting互斥分开；后三项bridge阶段只是actions内的细分，不可再次加总，也不代表覆盖callback/factor/residual的所有传输。逐variant独立进程、每case首调用及两次warm、重建初始数值metric/factor、保留失败returncode/rows，符合此项对照范围。失败后的warm中位数是有条件统计，必须同时给出尝试/通过分母。

首批归档防护验证各variant snapshot等于各自environment和当时current runner，最终CPU核验另外确认全部等于study起始hash。后续profile已加入该起始hash的逐variant断言，以排除variant之间改代码。45秒预算是callback软预算，不是进程硬超时；达到预算、评价上限或native停止但未过物理门，均不得参与“成功重建”的加速统计。

## 固定白化矩阵的额外探针边界

当前每个直接/缓存Jacobian动作仍包含白化三角解及prior动作。可用device-resident microbenchmark分别测decode三角解、prior GEMV、物理图像JVP/VJP、完整direct/cached动作，定位单次成本；这些独立调用有不同fusion/中间值复用，不能把它们机械相加当作完整动作时间，也不能把microbenchmark速度直接外推为fit加速。

预先显式构造\(T=L_0^{-T}\)，再将`solve_triangular(L0.T,z)`改成`T @ z`，是可以测试的固定坐标实现。应以矩阵右端`solve_triangular(L0.T, I)`构造T，保留原物理prior与目标；不是取\(H_0^{-1}\)，更不是令prior为单位矩阵。[JAX triangular solve接口](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.solve_triangular.html)。它增加每fit一次构造成本和约90.15MB的3357² float64矩阵，必须计入总成本及存储。

有限精度显式T与逐次三角solve不完全相同，病态矩阵可能放大误差，不能仅凭未抛异常判定准确。[NumPy官方逆矩阵精度说明](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html)。应核对\(L_0^TT\approx I\)、方向/adjoint和物理解码误差。只要T保持满秩、decode/AD使用同一T且最后在物理q上验收，它仍参数化同一个F；精确白化程度影响条件数，而非物理目标本身。浮点重排可能改变bilinear分支及外迭代/动作数，因此最终速度归因同时需要每动作成本、总动作数和完整fit验收。

后续`--whitening explicit`实现已只读核对：在实例首次trace前统一decode，单独保留metric Cholesky，由其转置与I求T；残差、导数、callback和终点使用同一T。initial directional expected正确使用\(T^Tg_q\)，物理初始metric/阈值、prior及最终物理证书未变。每fit重建T，构造时间包含在factor/setup/total，`explicit_whitening_seconds`是嵌套子计时，不应再次加总。默认solve路径保持原实现。

## 六方案实测与独立核验

已CPU核对 [gn_performance_study完整summary](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/gn_performance_study/summary.json)：六方案各有subhalo/control首调用和两次warm，共36个终点。所有runner snapshot都等于study起始hash，记录的依赖hash一致；原config、data/noise/mask、initial joint values、源轴、势节点完全一致。summary/diagnostics/optimizer相符，保存endpoint与fields相符。36个终点均通过fresh物理门和独立AD/解析门，均保留native success=false/status=-2、physical_gradient_converged及owned callback副本。cached各方案的factory/push/pull/matmat缓存计数均为1，没有因点变化重复编译的证据。

| 方案，其他条件同baseline | Subhalo warm中位数/s | Control warm中位数/s | 每case通过数，含首调用 |
| --- | ---: | ---: | --- |
| baseline：direct/every/LSMR 1e-10 | 8.8814 | 2.2310 | 3/3 |
| candidate：仅候选调度 | 6.5073 | 1.2493 | 3/3 |
| tol1e8：仅LSMR 1e-8 | 8.0082 | 2.1889 | 3/3 |
| tol1e6：仅LSMR 1e-6 | 6.9090 | 2.1495 | 3/3 |
| cached：仅缓存当前线性化 | 7.8859 | 2.2229 | 3/3 |
| combined：candidate/cached/LSMR 1e-8 | 4.7598 | 1.1953 | 3/3 |

候选调度把subhalo的当前对角检查从平均23.5次减到6次、control从9次减到2次，callback均值分别从3.339降到0.885秒、1.283降到0.297秒；这直接解释其收益。仅放宽到1e-6时，subhalo平均Jacobian动作数从3203减到1821，动作时间4.684降到2.678秒，外迭代仍约24次；主要省去的是内层精度成本。仅cached的subhalo动作均值从4.684降到3.602秒，额外线性化构造0.026秒，control则因原动作成本较小而几乎不改善总量。

组合方案相对本次fresh baseline约1.866/1.867倍加速。其subhalo仍有2668次Jacobian动作、耗时约3.011秒，callback约0.887秒；control动作约0.160秒、callback约0.300秒，而初始准备仍约0.54秒。它没有追平此前L-BFGS成功样本；不同方法的审计/停止开销及此前L-BFGS失败记录仍须保留。所有改善仅适用于此GPU、配置、两次warm和该实现，不是一般算法优劣或可靠性估计。三项措施没有测试所有组合，不能称当前combined为最优配置。

## 设备常驻组件与桥接探针

[micro结果](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/gn_performance_micro/results.json) 使用baseline/subhalo/repeat_1终点，endpoint和runner hash、保存的microbench快照、三组median和矩阵bytes已CPU核对。脚本先warmup，再随机调用次序，每组40次、共三组；float64输入始终在设备，逐次同步。它测的是该点和固定向量的设备调用墙钟。

| 动作，中位数/ms | Forward/JVP | Transpose/VJP |
| --- | ---: | ---: |
| 固定白化三角解 | 0.424 | 0.635 |
| 显式T GEMV | 0.133 | 未单独测 |
| 物理图像导数 | 0.120 | 0.231 |
| 完整direct augmented导数 | 1.315 | 1.417 |
| 完整cached augmented导数 | 0.829 | 1.206 |
| cached + 显式T augmented导数 | 0.350 | 0.474 |

完整prior Cholesky GEMV另测约0.160ms。白化三角解比物理图像导数本身更贵，支持进一步测试显式T，而不是认定主要成本来自求图像导数。T首次构造墙钟0.18654秒，可能含该形状编译/初始化；其归一化inverse residual约4.09e-15、单向量相对误差1.96e-15，direct/dense动作断言通过。该探针本身不证明所有点/控制case的稳定性或完整fit收益；下节另给出完整配对试验。

[bridge结果](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/gn_performance_bridge/bridge/summary.json) 每case首调用和一次warm，共4次均通过，snapshot匹配study起始hash。单次warm的subhalo动作4.7954秒，分为H2D 0.2281、device-call 4.4174、D2H 0.1484秒；control为0.2443秒，分为0.01146、0.22525、0.00753秒。传入/传出约占动作8%，其余大多位于含dispatch/wait的device-call；不能把约92%称为纯GPU kernel，也不能把Jacobian总成本归为数据搬运。额外同步后的总fit约8.947/2.228秒，与baseline接近，但此处各只有一个warm样本。

## 显式白化的完整配对结果

[gn_performance_whitening](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/gn_performance_whitening/summary.json) 用更新后同一runner重新运行四方案，各有每case首调用和两次warm，共24个终点。已CPU逐一核验study起始hash/全部snapshot/依赖hash、原config、data/noise/mask、初始状态、源轴及势节点、summary/diagnostics/optimizer和endpoint/fields。24次均通过fresh物理门及独立AD/解析门，保留正确native状态与停止副本；显式T方向变换检查最大相对误差约3.97e-13，独立梯度GN metric差最大约4.03e-13。

| 本批配对方案 | Subhalo warm中位数/s | Control warm中位数/s | 每case通过数，含首调用 |
| --- | ---: | ---: | --- |
| fresh baseline：direct/every/1e-10/solve | 8.9585 | 2.2396 | 3/3 |
| explicit-only：其余保持baseline | 6.4151 | 2.1820 | 3/3 |
| combined6：cached/candidate/1e-6/solve | 3.7928 | 1.1675 | 3/3 |
| fast：combined6加explicit | 2.7114 | 1.1575 | 3/3 |

fast相对本批baseline约3.304倍/subhalo、1.935倍/control。其subhalo成本约为setup 0.601、Jacobian动作0.957、线性化0.019、callback 0.881、终点0.148秒，其余约0.106秒。相比combined6，动作成本由2.064降至0.957秒，平均动作数1819/1808接近，说明显式T主要降低单动作成本；每fit额外重建T约0.0666秒已计入setup，90.16MB额外矩阵也如实记录。

control的combined6与fast分别1.1675/1.1575秒，只差约0.010秒、约1%，在两个warm样本上不足以证明显式T有稳定收益：动作只省约0.056秒，接近新增T构造约0.0665秒。其主要收益来自候选调度和减少内层动作，源于大量动作时的白化改进不能直接推广到较便宜case。

这24次与原dense LM的最大F差约1.65e-5，最大图像差RMS/noise约1.99e-5，满足相同物理质量门。fast仍比既有L-BFGS成功warm样本约1.10/0.70秒慢；后者存在保存的subhalo失败样本，且此处包含逐轮外部门和更多初始audit，比较须保留这些实现/验收口径。当前证据支持GN原型有可验证的性能改进空间，并未证明已找到最优策略或长期收敛率。首批36次与本批24次来自多个配置及重复的同一个mock，不是60个独立数据集。

## 精确GN对角的设备分组实现

继续优化验收本身可以不依赖候选筛选。对当前物理q，令\(A=D[f(q)/\sigma]\)，精确GN对角为

\[
H(q)_{ii}=\|Ae_i\|_2^2+Q_{ii}.
\]

新增`compiled_diagonal`在当前点只`linearize`一次，然后`lax.map`处理210组、每组16个one-hot方向，立即将16幅导数图归约为16个平方范数，切去末尾3个padding方向后加原dense Q对角。它完整计算GN对角；物理目标仍保留完整Q，取对角时只需其Qii。物理门仍使用q的曲率。JAX官方将`lax.map`列为减少内存、避免Python循环的实现方式，[lax.map](https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html)；越界one-hot为零，故padding可安全截去，[one_hot](https://docs.jax.dev/en/latest/_autosummary/jax.nn.one_hot.html)。

源码只读核对：one-hot只在block内生成，每组输出16个norm，没有堆叠整个m×n Jacobian或n×n方向矩阵。prior Q在factor阶段构建一次并提取对角；初始compiled值与原H0对角比对，callback和最终证书都通过同一current_diagonal选择，独立endpoint仍使用旧AD panel与解析实现。新callback中value/gradient、screen、diagonal子计时均嵌套于external-gradient总计，不可重复相加；对所有新配对variant一致增加gradient同步，计时内部可比较。compiled_every逐轮精确检查可避免candidate调度的启发式边界，实际成本与验收仍由完整对照确认。

[最后配对结果](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/gn_performance_diagonal/summary.json) 的18个终点已CPU核验，原config/data/noise/mask/初值/几何、全部study/snapshot/依赖hash、summary/诊断、endpoint/fields一致，均通过独立物理门，native状态和owned callback副本正确。active缓存大小均为1；compiled初始对角相对误差最大1.37e-15，独立AD/解析梯度GN metric差最大4.44e-13。

| 同为cached/explicit/LSMR 1e-6 | Subhalo warm中位数/s | Control warm中位数/s | 每case通过数，含首调用 |
| --- | ---: | ---: | --- |
| fast：candidate/panels | 2.7417 | 1.1576 | 3/3 |
| compiled_every：every/compiled | 4.9381 | 1.9820 | 3/3 |
| compiled_candidate：candidate/compiled | 2.6385 | 1.1906 | 3/3 |

compiled将单次精确对角从约0.143秒降到0.113秒，但每fit初始一致性audit使setup由约0.61增加到0.72秒。compiled_every的subhalo实际检查26次、对角累计2.941秒；compiled_candidate检查6次、累计0.680秒。两者动作成本仍约0.99/0.98秒。完整逐轮检查因次数更多而显著较慢；compiled_candidate对subhalo仅快约3.8%、control反而慢约2.8%，两次warm不足以证明这个可选后端有稳定端到端收益。不能因为GPU循环或精确公式就预先断言性能改善，也不能把外迭代从24变26的影响归为纯单动作优化。

还按NPZ独立重算了实际恢复质量：subhalo峰0.303596–0.303601，全部仍在(0.75,0.075) arcsec；source relative L2约0.070295–0.070296、source RMSE约0.024576，source相对原LM解差约2.08e-5–2.24e-5。Control kappa RMS约0.000532926、峰约0.00150639，source relative L2约0.0517100，相对原LM源差约8.63e-7。它们与保存recovery元数据一致，支持这些性能对照维持原重建质量；峰压低和先验影响仍存在。

## 推荐配置的观测初始化与失败回归

[推荐fast配置的source-only验证](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/gn_perf_source_only/summary.json) 使用cached/candidate/LSMR 1e-6/explicit/panels。原config、data/noise/mask、源轴/势节点、runner及依赖hash、owned endpoint/fields与独立AD/解析证书已CPU核验。源初值从观测的零势source-only MAP求得，确实不同于truth节点值：初始relative L2约0.0962868/subhalo、0.0510371/control；初始势为零，source方程相对残差约7.7e-16。记录`source_truth_used=false`，同时仍保留`macro_fixed_to_truth=true`的边界。

| source-only终点 | Subhalo | Control |
| --- | ---: | ---: |
| F | 11734.8265412363 | 11404.4436554436 |
| fresh gamma / 原政策阈值 | 4.9964004e-4 / 5.0016816e-4 | 9.6533461e-6 / 3.0000041e-5 |
| 与原LM的F差 | +2.75e-5 | -2.14e-7 |
| 图像差RMS/noise | 2.77e-5 | 1.09e-6 |
| 最终源relative L2 truth误差 | 0.0702922 | 0.0517100 |

两次均通过约定门及独立证书，保留native success=false/status=-2、physical_gradient_converged与匹配的独立停止副本。Subhalo峰0.303554仍在(0.75,0.075) arcsec；control kappa RMS约0.000532962。Subhalo在原允许阈值附近停止，与部分truth初值终点存在小幅F差，结论限于该门下的重建一致性。该验证支持推荐配置不依赖truth-node源初值；初值不同，本节不计算相对truth cohort的加速比。

[评价预算回归](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/gn_perf_eval_budget/summary.json) 的max_nfev=2保留native status=0、success=false、stop_reason=null及评价上限message。保存的native返回点有限，末次history与终点F相符，fresh gamma约0.10163明显高于1.00102e-6，独立解析门同样失败，accepted=false；snapshot和依赖hash已核对。未把评价上限当作callback异常或收敛。

[墙钟预算回归](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/gn_perf_wall_budget/summary.json) 的max_seconds=0.0001保留native status=-2、success=false、callback_wall_budget与owned callback停止副本，返回native参数与其相符。F约0.485948、fresh gamma约0.16567远高于1.00102e-6，独立解析门同样失败，accepted=false。snapshot/依赖hash、有限终点、末次history一致性已CPU核验；软预算在外迭代callback检查，不能解释为0.0001秒硬超时。

另只读审查 [CPU跨点数学检查](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/audit_gauss_newton_operators.py:20)：两toy × 两白化 × 三个变化点，覆盖augmented objective、物理/白化梯度链式、cached/direct JVP/VJP、adjoint、J.T@r与scalar gradient一致性、compiled/panels对角。这些是有意义的数学一致性核验，显式value_grad映射也被覆盖；不构成性能或全局收敛率测试。本审查未另运行JAX任务。
