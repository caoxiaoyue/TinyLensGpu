# 论文迭代法与SciPy/JAX Gauss–Newton接口核对

2026-11-03。只读主源研究；不安装依赖、不运行拟合或GPU任务。数学依据为本地 [2510.07382v1.pdf](/home/xycao/workspace/PT_dev/paper_pdf/2510.07382v1.pdf) §1.4、式(13)–(18)，接口依据为已安装SciPy1.17.1源码及官方文档。

## 论文的式(18)是正则化Gauss–Newton曲率

论文固定宏模型，令联合状态 \(q=(s,\psi)\)，当前线性化列块为 \(L_{GI}=[L\mid-LG_sG_\psi]\)。其式(18)使用

\[
[(DL_{GI})^TC^{-1}(DL_{GI})+R_{GI}]\Delta q
=-(DL_{GI})^TC^{-1}(DLs-d)-R_{GI}q.
\]

这对应当前正则化非线性最小二乘的Gauss–Newton步。作者将它称作Newton iteration；按曲率表达式严格分类，应称正则化Gauss–Newton。[论文HTML](https://arxiv.org/html/2510.07382v1#S1.SS4)。

一般写 \(e=f(q)-d\)、\(F=\tfrac12e^TWe+\tfrac12q^TQq\)，则

\[
\nabla F=J^TWe+Qq,\qquad H_{GN}=J^TWJ+Q,
\]
\[
H_{full}=H_{GN}+\sum_i(We)_i\nabla^2 f_i.
\]

式(18)没有最后这一残差加权的模型二阶导数项，因此不是完整Newton Hessian。其正则化作用于累积q，右端有\(-Qq\)，不是只惩罚本轮增量。

该节描述从联合q=0开始，反复全步更新\(q\leftarrow q+\Delta q\)，实际使用20次内迭代，再在外层优化正则化强度；该段没有描述LM阻尼或line search。不能把论文里固定次数/外层evidence搜索与本次固定强度MAP试验混为一谈。本次沿用既有mock的truth-node初值以隔离求解器对照，也不是复现论文的零初值。论文数学结构与当前图像原型相通，但数据算子和具体prior不是同一实验。

本地 [参考 iterative.py](/home/xycao/workspace/PT_dev/lensing_potential_correction/potential_correction/iterative.py:259) 则显式构建\(J^TC^{-1}J+R\)、再加\(\mu I\)，重算候选非线性F并只接受下降；这是阻尼GN/LM。调整mu会改变步长和方向，并非沿固定方向只缩放alpha的常规line search。当前TinyLensGpu也采用GN曲率并用对角metric阻尼：[当前backend](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py:69)。

## 现成SciPy路径

建议的现成接口是：

```python
scipy.optimize.least_squares(
    residual_numpy, z0,
    jac=jacobian_linear_operator,
    method="trf", tr_solver="lsmr", loss="linear", x_scale=1.0,
    tr_options={"regularize": True, "atol": 1e-8, "btol": 1e-8},
    callback=callback,
)
```

容差这里只是可记录的起始配置，不是已验证的足够精度。`trf+lsmr`是带信赖域保障、以GN模型为基础的现成非线性最小二乘策略；它不是论文式(18)那样无条件走完整GN步。LSMR产生近似GN方向，TRF在梯度/GN方向张成的二维子空间解信赖域子问题，按实际/预测下降调整半径。[官方least_squares说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)、[已安装trf_no_bounds](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/scipy/optimize/_lsq/trf.py:415)。

只提供scalar objective和\(\nabla F\)不够定义GN曲率；需要完整残差向量及其Jacobian或等价JVP/VJP。`least_squares(method="lm")`仍要求dense Jacobian并进入MINPACK路径，不能使用下面的LinearOperator替代。[已安装接口约束](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/scipy/optimize/_lsq/least_squares.py:985)。

## JAX导数与当前Jacobian的无矩阵接口

保留原完整Q，令\(Q=L_Q L_Q^T\)，使用

\[
r(z)=\begin{bmatrix}(f(q_0+Tz)-d)/\sigma\\L_Q^T(q_0+Tz)\end{bmatrix}.
\]

若固定初始曲率\(H_0=L_0L_0^T\)，取\(T=L_0^{-T}\)只改变坐标；prior仍作用于累积物理状态，不可替换成\(\|z\|^2\)。JAX给出

- `matvec(v) = jax.jvp(residual, (z,), (v,))[1]`，形状m；
- `rmatvec(u) = jax.vjp(residual, z)[1](u)[0]`，形状n。

据此构造`LinearOperator((m,n), matvec=..., rmatvec=..., dtype=np.float64)`，作为`jac(z)`返回。完整200²的Cholesky-prior残差为m=25373、n=3357；无需形成约681MB的当前dense Jacobian，也不需要原lmfit的CPU dense QR。[LinearOperator官方定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html)、[JAX JVP](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html)、[JAX VJP](https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html)。

实现应显式处理输入`(n,1)`/`(m,1)`并ravel；TRF的`J.dot(S)`会触发matmat，默认实现可能传列向量。每个返回的operator应保存其线性化z的独立副本，不能让旧operator读到后一次jac调用更新的点；JIT函数身份保持稳定，point/factors作为动态参数，避免每次matvec重编译。实际初始H和prior Cholesky仍为dense，所以这里“无矩阵”只指当前迭代Jacobian/LSMR动作，不表示整个方法已完全线性存储。

`x_scale="jac"`不支持Jacobian为LinearOperator；固定白化后使用1或明确正数组。`loss="linear"`保留原F；改robust loss会连prior残差也一并改变，不是单纯换优化器。[已安装检查](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/scipy/optimize/_lsq/least_squares.py:997)。

## 数值保障、终止与计数

- `tr_options["regularize"]`默认True，用于LSMR/GN子问题的临时算法regularization；SciPy据当前梯度/信赖域计算reg_term，再传`damp_full`给LSMR。它不修改fun中的物理Q，不能把它当成新增势先验或从物理penalty扣掉。设False也仍有TRF信赖域保障，不会变成无条件全步GN。[已安装TRF](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/scipy/optimize/_lsq/trf.py:475)。
- `lsmr`内层`atol/btol/conlim/maxiter`控制近似线性求解；外层`max_nfev`控制残差调用，不是LSMR迭代预算。除nfev/njev外应记录JVP/VJP/matmat次数及时间，避免把许多GPU/CPU同步藏在一次“Jacobian evaluation”里。[LSMR官方定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html)。
- TRF允许`ftol=None`、`xtol=None`关闭相应停止条件，但至少一个容差必须高于机器epsilon。其native gtol作用在优化坐标/边界尺度，不能代替本项目原物理GN对角梯度门。[已安装容差检查](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/scipy/optimize/_lsq/least_squares.py:120)。
- 已安装1.17.1支持TRF callback：`callback(x)`，或参数名严格为`intermediate_result`的`callback(intermediate_result)`。后者具有x/fun/cost/nit/nfev；不要假定其包含jac或grad。callback抛`StopIteration`后status=-2、success=False应如实保留，再独立检查原物理门，区分主动外部收敛与预算退出。[callback封装](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/scipy/optimize/_optimize.py:87)。
- **不能沿用lmfit的return True停止语义**：实际`_call_callback_maybe_halt`忽略正常返回值，只捕获`StopIteration`。[已安装停止实现](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/scipy/_lib/_util.py:988)。保留已检查点的独立副本，并在API返回后重算AD及解析物理证书，延续此前的透明验收政策。

该路线解决的是将GN线性动作与CPU dense QR分离，仍会有SciPy/LSMR的CPU向量控制和JAX设备传输。实际收敛与速度须由原完整subhalo/control及相同停止质量的对照给出，不能凭算法名称预先判优。

## 原型与结果审查

已只读审查 [run_gauss_newton.py](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/run_gauss_newton.py:1)：

- 完整prior Cholesky、累积物理q、固定初始metric坐标方向正确；JVP/VJP与batched matmat的形状/adjoint方向正确，operator拥有各自的线性化点，JIT函数身份稳定。
- 显式TRF+LSMR、linear loss、无bounds、x_scale=1、algorithmic regularize与物理prior分开，符合上面的现成GN路线；未形成当前dense Jacobian或进入CPU MINPACK QR。
- callback参数名和`raise StopIteration`正确，保存独立点；native status/success原样记录，status=-2的物理门停止与预算停止区分。独立AD/解析物理门都通过后才接受，未拿native optimality代替原项目证书。
- 时间从fit入口起算，包含metric、两项Cholesky、目标/方向/adjoint初始audit、全部TRF/LSMR动作及CPU/GPU传输、callback检查、最终证书和raw products。排除mock/几何准备、fit返回后的文件导出以及额外独立解析审计；API内进度打印计入。该原型与其他方法的比较应使用相同或明确标注的时间口径。

SciPy1.17.1无边界TRF在callback停止时的`result.optimality`可能沿用该轮顶部的旧g_norm，而当前g在接受新点后才刷新；此字段按native值保留即可，不能当作这里独立重算的终点物理梯度。

已CPU核对 [coupled smoke summary](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/gauss_newton_coupled/summary.json)：首调用和两次重复均接受，F约0.446379594795874，原物理gamma约7.48e-7、阈值1.001e-6，独立AD/解析gate均过，snapshot hash匹配。三次均保留native success=False/status=-2和physical_gradient_converged，数学方向及adjoint audit一致。每次15次operator构造对应约300次JVP、290次VJP和14次matmat，不能把15次jac计数当作全部导数工作量；这只是小型非线性接口验证，不能替代完整200²结论。

## 原始200²、truth-node初值的完整结果

已CPU核对 [完整summary](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/gauss_newton_full_truth/summary.json) 的六个终点（每case首调用+两次重复）。原始科学JSON完全相同，归档runner snapshot SHA256匹配environment；各NPZ的图像/noise/mask、源轴、势节点与保存dense LM baseline一致，3357维float64，truth-node源初值和零初始势保持，endpoint与diagnostic_fields一致，summary与每次diagnostics一致。

| 原200²，固定原强度 | Subhalo | Control |
| --- | ---: | ---: |
| 通过外部物理门和独立AD/解析审计 | 3/3 | 3/3 |
| native status / success | -2 / false | -2 / false |
| stop_reason | physical_gradient_converged | physical_gradient_converged |
| 最终物理gamma范围 | 3.62243e-4–4.90457e-4 | 约2.99609e-5 |
| 原物理阈值 | 5.0097607e-4 | 3.0055413e-5 |
| F范围 | 11734.82651923–11734.82653027 | 11404.44365583 |
| 相对原dense LM F差 | +5.44e-6至+1.65e-5 | +1.70e-7 |
| 图像相对原dense LM差，RMS/noise | 1.38e-5–1.99e-5 | 6.94e-7 |
| 首调用fit total | 14.885 s | 5.176 s |
| 两次复用kernel的fit total | 8.793 / 9.066 s | 2.215 / 2.238 s |

全部为独立保存的callback点，且返回native参数与捕获副本一致。原样保留SciPy的status=-2和success=False，是因为达到约定物理门后主动触发公开callback停止；不能称为native gtol成功。六个终点AD/解析梯度在同GN metric下的差约7e-14–4e-13，远小于阈值，control接近阈值的通过也有充分数值余量。

Subhalo峰值约0.303596，仍在(0.75,0.075) arcsec，恢复和原LM一致，离散真峰仍为0.631435；这不是消除了正则化峰压低。Control all-core kappa RMS约0.000532932，与原control一致，没有因换求解器引入明显新的伪信号。这些重建检查支持同目标、同先验下的实用等价解，不是全局最优或检测显著性结论。

每次都重新计算初始metric和Cholesky，没有缓存其数值；计时包括前述全部导数动作、外部callback证书和终点products。Subhalo每次约1590–1650次matvec、1560–1630次rmatvec，迭代23–24轮；control为77次matvec、69次rmatvec和9轮。Warm callback证书累计约3.3秒/subhalo、1.3秒/control，已计入总量；它是当前原型实际成本的一部分。

与此前同硬件、同原问题的 [dense LM warm记录](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/timing_full/lm_subhalo.json)、[control记录](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/timing_full/lm_control.json) 33.802/13.917秒相比，新两次warm约为3.8/6.3倍加速。这是已有批次与本次两次warm样本的scope已说明比较，不是新的严格配对benchmark；新原型含额外初始audit和逐轮证书，也不能把所有耗时差归结为一个算法性质。

本次结果支持使用现成SciPy TRF/LSMR与JAX Jacobian动作替代手写非线性控制，并避免lmfit CPU dense QR瓶颈。它仍慢于此前L-BFGS的成功样本；此前L-BFGS有保留的物理门失败，而这里六次成功也只代表本次小批次，不能据此估计长期可靠性。该实现是带信赖域保障的GN方案，保留dense初始白化准备，不是无保护GN全步、完整Newton、纯GPU求解器或论文原始零初值/固定20步流程的完全复现。

## 不用源真值的初始化补充与预算失败路径

已CPU核对 [source-only summary](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/gauss_newton_full_source_only/summary.json)：原科学JSON、图像/noise/mask、源轴、势节点及truth字段与原baseline/truth cohort一致，runner归档和记录的全部依赖SHA256匹配。初始化采用零势修正下的signed source-only MAP解，使用观测、固定宏模型和原源prior，没有读取源真值；[initializer代码](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/run_scientific.py:293) 和保存的`source_truth_used=false`一致。两case初始源与真值的relative L2差分别为0.09629和0.05104，初始势全零，source方程相对残差约7.4e-16。宏模型仍固定为真值，因此不能称为无需任何真值的完整观测推断。

| 原200²，source-only初值 | Subhalo | Control |
| --- | ---: | ---: |
| accepted，独立AD/解析门均过 | true | true |
| F | 11734.82651640255 | 11404.44365555967 |
| fresh物理gamma | 3.3658233e-4 | 1.4387624e-5 |
| 该初值对应原政策阈值 | 5.0016816e-4 | 3.0000041e-5 |
| 相对原dense LM的F差 | +2.62e-6 | -9.78e-8 |
| 图像差RMS/noise | 1.20e-5 | 9.30e-7 |

阈值继续使用原政策`atol + 1e-8 * initial_gamma`，所以改变初值后与truth cohort的数值略有不同。两终点均保留native success=false/status=-2、physical_gradient_converged和owned callback副本；native参数与副本相同，endpoint与diagnostic_fields相同，独立AD/解析梯度的GN metric差分别为9.50e-14和3.05e-13。Subhalo峰约0.303580，仍在(0.75,0.075) arcsec；control kappa RMS约0.000532951，重建与原解相符。这支持结果并非依赖truth-node源初值。本批次准备期间有并行绘图导出，**source-only耗时不用于任何benchmark或加速比**。

另已核对两个小型失败回归的归档hash、summary/diagnostics/optimizer一致性、有限endpoint和独立梯度证书：

| 预算路径 | native状态 | fresh gamma / 阈值 | 保存与验收 |
| --- | --- | --- | --- |
| [max_nfev=2](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/gauss_newton_eval_budget/summary.json) | status=0，success=false | 0.10163 / 1.00102e-6 | native返回点，accepted=false |
| [max_seconds=0.0001](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/gauss_newton_wall_budget/summary.json) | status=-2，success=false | 0.16567 / 1.00102e-6 | callback_wall_budget，owned副本，accepted=false |

两次独立解析门同样失败，未把下降或callback中断当成收敛；保存状态与main的失败退出1分支一致。墙钟预算从求解API阶段开始、在外迭代callback检查，不限制此前准备，也不是硬进程超时。其返回native点与拥有的停止副本一致，末次history目标与保存终点目标一致。
