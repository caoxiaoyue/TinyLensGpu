# lmfit Levenberg–Marquardt：接口与科学验收核对

2026-11-03。只读研究；不改库/runner、不安装包、不运行GPU任务。已读取 `tinylens_gpu` 中的 lmfit 1.3.4、SciPy 1.17.1 源码，并做一个单参数CPU接口检查。

## 目标与接口

应使用 `lmfit.Minimizer(...).leastsq(Dfun=..., col_deriv=...)`，或 `minimize(..., method="leastsq")`。这里 `leastsq` 对应 SciPy/MINPACK 的 LM；`least_squares` 默认是另一求解方法，不能把默认值混称为相同LM。[lmfit官方说明](https://lmfit.github.io/lmfit-py/fitting.html)、[官方Jacobian示例](https://lmfit.github.io/lmfit-py/examples/example_fit_with_derivfunc.html)。

LM接口需要**固定长度残差向量**，不接受原 scalar penalty 作为替代。对当前物理变量 \(x=(s,p)\) 及精度 \(Q=L_Q L_Q^T\)，一个直接的等价残差是

\[
r(x)=\begin{bmatrix}(d-f(x))/\sigma\\L_Q^Tx\end{bmatrix},
\qquad F(x)=\tfrac12 r(x)^Tr(x).
\]

Q必须使用现有全部源平滑、势完整四阶差分及所有势节点的幅度先验，不能新增数值ridge、遗漏amplitude项或改变差分单位。prior残差作用在**累积物理x**，不是\(x-x_0\)。若保留坐标预条件 \(x=x_0+Tz\)，完整残差和Jacobian都必须在该映射下计算；不能把prior残差直接替换成z。当前目标/精度依据：[prior_matvec](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/_potential_correction_common.py:133)、[dense_prior](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/_potential_correction_common.py:151)。

用户 residual 与 `Dfun` 均接收 `Parameters, *fcn_args, **fcn_kws`，不是裸NumPy参数向量。参数顺序必须与 `result.var_names` 的自由变量顺序一致。原问题为有符号无约束优化，参数保持无bounds、无expr、vary=True；否则lmfit还会进行边界内外坐标变换。[已安装 `_jacobian`](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/lmfit/minimizer.py:556)。

| `col_deriv` | 用户Jacobian形状 | 本例Cholesky-augmented residual |
| --- | --- | --- |
| False | (n_residual, n_parameters) | (25373,3357) |
| True | (n_parameters, n_residual) | (3357,25373) |

完整导数为 \(J_{r,z}=J_{r,x}T\)；残差图像项使用data-model时导数有负号，prior部分为\(L_Q^TT\)。`col_deriv=True`只是转置存储约定，不会替你计算导数。lmfit在False分支执行`jac *= grad_scale`，因此从JAX转换来的只读`np.asarray`应改为可写拷贝，或明确使用True方向并核验维度。[SciPy接口语义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html)。

## 容差与调用预算

- 使用lmfit的 `max_nfev` 明确限制残差调用次数，不把它称为LM迭代数；Dfun调用次数另外计数。lmfit默认预算为2000×(nvarys+1)，此例极大。其wrapper有自己的残差计数/abort，再向SciPy传`maxfev=2*max_nfev`；不要凭SciPy默认估算总预算。[已安装 leastsq](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/lmfit/minimizer.py:1609)。
- MINPACK的`ftol`是平方残差和的相对减小，`xtol`是参数步长，`gtol`是残差与Jacobian列的正交性条件；它们都不等于原实现的物理GN缩放梯度阈值。`diag`必须是正数，定义变量尺度；它只改变求解尺度，不应进入物理penalty。[SciPy leastsq说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html)。
- 保留原始subhalo/control的外部物理梯度策略，native成功后重新计算当前点证书；未过证书的终点如实保存为诊断。`lmfit_result.chisqr = ||r_aug||² = 2F`，不能直接与旧F列比。由于prior行是伪观测，lmfit自动给出的redchi/AIC/BIC也不能直接当成原物理模型的统计推断结果。

## 关键陷阱：`calc_covar=False`不能关闭leastsq协方差

官方文档明确将`leastsq`和`least_squares`排除在`calc_covar`开关之外。已安装1.3.4源码确认：

1. lmfit leastsq强制`full_output=1`；SciPy在成功返回时由最终QR的R构造`cov_x`，包含triangular inverse与dense乘法。[已安装SciPy](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/scipy/optimize/_minpack_py.py:478)。
2. lmfit随后转换协方差到外部参数，并调用 `_calculate_uncertainties_correlations`；本分支没有检查 `self.calc_covar`。[lmfit调用点](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/lmfit/minimizer.py:1713)。
3. 后处理为所有自由参数构造stderr、相关系数字典，并可能调用`create_uvars`。3357变量约有1127万个有向相关项，并包含进一步协方差处理。[后处理源码](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/lmfit/minimizer.py:808)。

`scale_covar=False`只取消redchi缩放，不跳过这些操作。传`full_output=False`也不是解决办法：lmfit随后固定解包5项返回值，会破坏接口。单参数CPU检查实际观察到：`calc_covar=False`、native成功时covariance仍存在，后处理hook执行一次；这与文档和源码一致。[官方calc_covar说明](https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.minimize)。

没有找到可公开、完整关闭此方法协方差的选项。可选择：

- **完整API基线**：保留真实lmfit调用，在后处理hook的计时wrapper中调用原实现，单独报告后处理和full API时间；从总时间减去该hook也仍包含SciPy的covariance构造，不能称为完全无covariance的MAP时间。
- **明确标注的实验适配**：局部`Minimizer`子类只override私有 `_calculate_uncertainties_correlations` 为 `result.errorbars=False; result.covar=None`，避免与MAP无关的相关dict/uvars处理。它不改残差、Jacobian和LM步长，但不是lmfit公开稳定开关，也仍保留SciPy的covariance工作。必须在输出说明适配范围，不伪装成默认完整API。

## 全尺寸的资源与速度含义

采用上述Cholesky prior残差时，Jacobian为25373×3357 double，单数组约681MB，MINPACK与lmfit可能再复制。JAX AD只负责计算模型/导数；lmfit/SciPy的LM线性代数仍在CPU，不能把调用JAX理解为整个LM进入GPU。

SciPy1.17.1的MINPACK `lmder`调用dense pivoted Householder `qrfac`，主要实现为C嵌套循环，不是把大QR交给GPU，也不是简单增加BLAS线程即可获得并行GPU性能。[对应版本源码](https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/optimize/__minpack.c)。据形状，QR工作量随 \(O(mn^2)\) 增长；这是复杂度判断，不是预报实测秒数。

CPU检查发现NumPy/SciPy各自的OpenBLAS默认均为32线程。应记录实际threadpool信息及CPU线程策略；covariance矩阵运算可能使用BLAS，过度线程竞争会干扰速度比较。避免在JAX编译/GPU程序活跃时另开重CPU/GPU任务。显式计入残差/Jacobian的GPU到CPU传输，保持可复用JIT对象，按同一scope计入正则化残差准备、参数尺度、求解、外部证书和raw products。

## 新runner与结果

已完成独立新文件 [run_lmfit.py](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/run_lmfit.py:1) 的只读科学/接口审查：

- `prior_chol.T @ values`作用于累积物理源/势，Cholesky方向正确。其图像残差采用model-data，与上面的data-model仅差该块整体符号，不改变平方目标；对应JAX导数保持一致。
- 初始坐标仍为原固定GN metric白化；`jacobian_block`直接对完整白化augmented residual作JVP，生成parameter-major的n×m数组，与`Dfun=jacobian, col_deriv=True`一致。残差、Dfun和iteration callback均收到正确Parameters签名，参数无额外bounds。
- 初始 `0.5*r.T*r` 与原标量F，以及随机方向的 `r.T*Jr*v` 与 `L_metric^-1*g` 的链式导数作独立核对；终点再次验证目标身份并重建当前GN对角物理证书。未用native success代替验收。
- 完整科学装配读取原JSON/相同生成器，检查保存的图像/mask/noise，并保留相同固定源/势网格、prior和truth/source-only初值标记；没有改系统库。
- total_seconds包含初始metric/两个Cholesky、参数对象、完整lmfit调用、真实covariance/uncertainty处理、终点证书和raw products。covariance transform及uncertainty hook只计时并调用原实现。`scale_covar=False`是显式报告设置，建议metadata注明；该运行保留完整后处理，而不是所有lmfit报告默认值原封不动。

审查发现并已在当前runner修复一个失败保存路径问题：lmfit1.3.4在callback主动abort后可能没有`result.chisqr`，直接读取会在保存预算超限诊断时抛异常。CPU最小例实际复现`aborted=True, success=False, hasattr(chisqr)=False`。当前版本使用可空native字段，并把已独立重算的augmented sum of squares另名保存，没有伪造native统计量；options也明确记录了`scale_covar=False`和callback预算不是硬超时。此前已启动进程使用其归档快照，不会因磁盘源文件更新而自动改变。

`max_seconds`是残差callback检查的wall budget，不能中断正在执行的单次CPU QR，也不限制随后的covariance/uncertainty阶段；因此它不是硬总时限。当前进度打印也在API计时中，后续速度解释应据真实scope说明。原LBFGS benchmark没有同样的协方差/相关后处理，full API时间可直接给用户，但不能把所有差异归因于LM迭代本身。

已只读核对 [coupled smoke记录](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lmfit_coupled/summary.json)：首调用及两次复用kernel的调用都通过原型物理阈值，F约0.44637959476212，scaled gradient约6.19e-9。另 [nodal smoke记录](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lmfit_nodal/summary.json) 的三次也全部通过，F约0.046095802148056，scaled gradient首调用1.93e-11、重复约1e-13。两个目录的runner snapshot SHA256均与各自environment一致。这些结果验证了接口、Jacobian方向和小型非线性问题的求解，原200²的完整证据见下文。

## 原200² control探针与预算失败的含义

已CPU核对 [probe_failure.json](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lmfit_full_control/probe_failure.json) 及归档快照hash。该完整control尝试使用3357变量、600秒残差callback预算；最后记录的F为11404.4436555（solver elapsed583.3秒），第9次AD Jacobian完成于586.3秒，生成该Jacobian约0.773秒。记录的相邻Jacobian间CPU阶段约83秒，说明主要成本在CPU MINPACK求解流程，而不是JAX导数生成。

随后触发callback预算abort，旧快照又在读取缺失的`result.chisqr`时抛出保存异常，因此**没有保存可验证物理终点或收敛证书**。这些数据只支持“完整API尝试超过预算、进度非常慢”，不能证明最后场未达物理阈值，也不能把接近基线的F当成已收敛，更不能给出成功拟合速度比。

当前guard/checkpoint修复已经通过 [abort smoke记录](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lmfit_abort_smoke/summary.json)：`aborted=True`、native chisqr为null、独立augmented norm保存、accepted=False。CPU复核其snapshot hash匹配environment。此修复只保障诊断保存，没有改变优化目标、Jacobian或LM更新。

## 原有64²校准配置的独立探针

当前runner已支持 `--configuration` 和 `--baseline`。代码确实读取指定JSON和对应baseline，校验源分辨率、完整三项prior强度及图像/noise/mask；不存在`case_iteration_options`时使用空字典默认，因此原64配置可装配。

已CPU核对 [calibration64 actual configuration](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lmfit_calibration64/original_configuration.json) 与仓库保存的 [原64配置](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/.scratch/potential-correction/output/iterative-scientific/calibration64/configuration.json) 完全相同，runner snapshot hash匹配environment；它使用真实既有校准问题，不是另造的较容易模型。

其64²、dpix=0.0375、PSF5、source49²，实际2252个数据、2401源节点、936势节点、3337联合变量；与原200²的采样/PSF不同。它可检验已知校准规模下的收敛与成本，但任何成功结果必须归属于64²配置，不能代替200²。

已CPU核对 [完整64 native summary](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lmfit_calibration64/summary.json)、两个diagnostics和NPZ：源轴、势节点、图像/noise/mask与既有64 baseline一致，truth-node初值完全相同、初始p为零；summary与各diagnostics一致，快照hash匹配。结果为：

| 64² native-return | Subhalo | Control |
| --- | ---: | ---: |
| lmfit native success / ier | true / 1 | true / 1 |
| 通过物理验收 | **false** | true |
| F | 1661.1273683881 | 1556.9688631401 |
| 原64 dense LM F | 1661.1273757723 | 1556.9688632243 |
| 最终GN缩放梯度 | 2.2950657e-4 | 7.5850777e-7 |
| 原阈值 | 3.0130142e-5 | 3.0021412e-5 |
| total，包括完整native后处理 | 534.067 s | 265.021 s |
| AD Jacobian次数 / 累计秒 | 29 / 17.794 | 15 / 9.299 |
| uncertainty/correlation hook秒 | 4.304 | 4.100 |
| 图像相对原64 LM差，RMS/noise | 2.65712e-5 | 2.50584e-6 |

独立AD/解析梯度都确认上述通过/不通过结论；同GN metric下导数差分别约3.28e-14、3.26e-13，排除了证书公式错误的解释。Subhalo虽然F低于原基线且图像接近，仍没有通过约定物理门，不能把native success当作有效收敛。其峰值0.2176603位于(0.75,0.075)，control all-core kappa RMS为0.001609720；这些属于64²结果，不能与200²的0.3036峰值混用。

总时间远大于已单列的AD与uncertainty hook时间，说明瓶颈主要在其余CPU MINPACK/强制SciPy covariance流程，不能把慢速主要归咎于JAX导数或最后相关系数报告。尚未测量同scope的旧64 LM或L-BFGS时间，因此这里不计算跨配置速度比。

## 可选原物理梯度callback停止变体

已只读审查新增 `--stop-on-gradient`（默认False）：它使用lmfit公开的`iter_cb`，每次残差调用时以当前物理x重新计算原F梯度和GN对角，只有原阈值通过且F不高于初始值才主动return True。主动abort原因与wall-budget原因分别记录，native `success/aborted/ier`不伪造；API返回后再独立重算证书。这作为**外部物理MAP停止变体**在数学上合理，不是MINPACK自身收敛成功的声明。[callback公开契约](https://lmfit.github.io/lmfit-py/fitting.html#using-a-iteration-callback-function)。

callback可能在一个trial点满足门时停止，并不要求该点已被MINPACK接受；只要明确保存该候选点、独立证明原目标/物理门和与基线的重建质量，仍可以作为本实验的MAP端点。它必须与默认native-return试验分开命名。因为主动abort会跳过native covariance/uncertainty阶段，其速度也不是完整native API统计后处理的速度；外部梯度检查成本应全部计入。

**重要的返回状态风险**：本环境lmfit1.3.4/SciPy1.17.1的纯CPU单参数LM测试三次复现：callback在最优x=2.0000000000000004、梯度约1.3e-15处return True后，library最终重算残差时的参数及`result.params`变为不相关的极小数，导致独立梯度回到6；callback中显式`np.array(..., copy=True)`保存的x仍正确。源码把`fvars`直接引用保存为`last_internal_values`，但这里未进一步修改/诊断其底层生命周期。

因此，首次触发physical或budget停止时必须独立copy参数向量到`stop['z']`或物理场，再使用该捕获数组做终点decode/证书/导出。不能依赖abort后的`result.params`，也不能使用会被library最终残差调用覆盖的`last['z']`。当前runner已按此实现，分别记录endpoint_source、native参数是否与捕获点相同；native flags保持不变，budget-only停止继续为失败。此前没有捕获数组的预算退出记录若返回了污染参数，不能当作可靠最后callback场；应使用已保存的Jacobian checkpoint另行审计，或仅作为预算失败记录。

这项保护不改LM算法，只保证公开callback主动停止时保留真正已检查的点。当前外部证书会拒绝污染参数，所以没有证据表明已发生误接受，但若不保护会丢失本来有效的callback点。

已只读核对保护后的两种smoke： [physical-stop nodal](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lmfit_external_smoke/nodal/repeat_0/diagnostics.json) 使用owned callback copy，native success=False/aborted=True原样保留，独立gamma=2.30e-8小于1.01e-6而外部接受；[budget-stop coupled](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lmfit_abort_active_smoke/coupled/repeat_0/diagnostics.json) 同样使用owned copy，但reason为callback_wall_budget且accepted=False。两种停止语义没有混同。

**额外的max_nfev角落**：已安装lmfit的`__residual`在调用user residual/iter_cb之前检查自己的评估预算。因此内部预算abort时，自己的`stop.reason`和`stop.z`可能仍为空，不能回退到native参数；它随后还可能以`result.aborted=True`再次调用最终残差/callback。CPU单参数非线性试验已复现：最后正常callback为x=1.414998429894803，abort后的native最终值变为无关极小数；若无条件“每次callback都copy”，这个最终回调还会覆盖好副本。

当前runner已保留**最近一次未处于native-aborted状态的完整callback**参数副本，并在`minner.result.aborted=True`的后续回调中不更新它、直接返回True。API返回后若native已abort且没有显式停止副本，使用这个最近正常callback副本；如果从未完成任何callback，就使用独立保存的初始z0。已只读核对guard确实位于正常callback-copy之前，且显式physical/wall停止副本优先。native fallback标记`native_abort_before_physical_stop`与独立endpoint_source，accepted保持False。CPU试验确认这一guard原理可保留正确副本；实现没有修正或伪造native状态。

当前main还要求独立重算的AD和解析GN梯度都通过原门后才保留accepted标志，已只读核对这一额外验收逻辑。它能发现错误终点，但不能替代上面的参数所有权保护。

最新实际 [max_nfev=4 regression](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lmfit_eval_abort_smoke/coupled/repeat_0/diagnostics.json) 已CPU核对：snapshot hash匹配；native参数确实与保留副本不相同，endpoint_source为`owned_last_completed_callback_before_native_abort`，F=0.4803813741120511与最后正常callback的F差约5.6e-17。native success=False/aborted=True/ier=-1、native chisqr=null均保留，reason为native_abort_before_physical_stop，accepted及独立物理门均False；说明该诊断回退实际保护了正确场，且没有把预算退出误报成收敛。

## 原200²、外部梯度停止变体：已保存subhalo终点

已CPU核对 [full-gradient subhalo diagnostics](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lmfit_full_gradient/subhalo/repeat_0/diagnostics.json)。原200² JSON未变，runner snapshot hash与environment相符；图像/noise/mask、源轴、势节点与原200² dense LM baseline一致。NPZ确认3357维float64、truth-node初值完全相同、初始p为零，endpoint.npz与diagnostic_fields的联合场一致，summary与diagnostics一致。

| 原200² subhalo，external-stop变体 | 测量 |
| --- | ---: |
| native success / aborted / ier | false / true / -1 |
| stop_reason | callback_wall_budget |
| endpoint_source | owned_copy_of_first_stopping_callback |
| native参数与捕获副本一致 | true |
| accepted / independent gate | **false / false** |
| F | 11739.649030480057 |
| F − 原dense LM F | +4.8225166943 |
| AD / 解析GN缩放梯度 | 0.528641943200396 / 0.528641943200420 |
| 原阈值 | 0.0005009760704720821 |
| fit total / lmfit API | 680.528 / 674.803 s |
| AD Jacobian次数 / 累计 | 9 / 7.308 s |
| 外部callback梯度检查累计 | 2.153 s |
| covariance/uncertainty后处理 | 0 s，预算abort后未执行 |

这次有可靠的已捕获物理终点和独立证书，明确未通过原物理门；不同于此前full-control因旧保存异常而无证书的记录。AD/解析梯度在相同GN尺度下差8.48e-14，失败不是导数审计误差或abort参数污染。记录仍保留恢复诊断：峰值0.3004518位于(0.75,0.075)，相对原LM图像差0.0127659 noise RMS；它们不能代替缺失的收敛条件。

600秒为soft callback预算，最后一次检查在solver elapsed674.66秒，包含一次无法中断的CPU QR导致的超出；不是已承诺的600秒硬进程时限。total从fit入口起算，**不含mock/几何准备、fit返回后的最终结果文件导出和额外独立解析梯度审计**；但包含setup中的目标/方向检查、Jacobian内的checkpoint写盘以及进度打印，不能声称排除了所有disk I/O。

原200² control同变体现已完成并经CPU核对：[control diagnostics](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lmfit_full_gradient/control/repeat_0/diagnostics.json)。同样确认原配置、snapshot hash、数据/几何、truth-node初值与零初始势、endpoint/fields及summary一致。

| 原200² control，external-stop变体 | 测量 |
| --- | ---: |
| accepted / 独立AD及解析gate | **true / true** |
| native success / aborted / ier | false / true / -1，未改写 |
| stop_reason | physical_gradient_converged |
| endpoint_source / native与捕获值相同 | owned callback copy / true |
| F / 相对原dense LM F | 11404.443655656942 / -5.07e-10 |
| AD / 解析GN缩放梯度 | 1.6124362709e-5 / 1.6124362784e-5 |
| 原阈值 | 3.0055413138e-5 |
| fit total / lmfit API | 421.528 / 418.871 s |
| AD Jacobian次数 / 累计 | 6 / 5.025 s |
| callback外部梯度检查累计 | 1.662 s |
| native covariance后处理 | 0，主动callback停止跳过 |

Control的独立AD/解析梯度差为2.56e-13（同GN尺度），非线性图像与原LM差1.85217e-6 noise RMS，all-core kappa RMS为0.0005329071，与原control重建相符。这是**外部物理门接受的MAP结果**；native仍如实报告主动abort，不能在表中改称lmfit自身成功返回。

最终证据支持：独立lmfit接口及完整prior/JAX Jacobian数学关系成立，小型coupled/nodal运行可通过；原200²下，外部停止的control取得与原LM一致的MAP，但一次调用约421.5秒，subhalo在soft预算后仍未通过物理门。两个完整case的AD Jacobian累计仅5–7秒且未执行covariance报告，因此本规模的主要成本位于当前CPU MINPACK线性代数流程，现有结果不支持以它作为更快、更稳定的直接替换。以上是本硬件、配置和具体后端的结果，不是对所有LM实现的判断。
