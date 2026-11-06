# JAX自动微分 + SciPy高斯–牛顿：独立实验

2026-11-03。新文件：[run_gauss_newton.py](run_gauss_newton.py)。使用已有 tinylens_gpu 环境（Python3.11.14、JAX0.7.1、SciPy1.17.1、RTX5080），无需额外安装包，未修改核心库。

**结论：这条路线在原200×200 Mock上可行。** 真值节点初值的6次运行和仅从观测恢复初始源的2次运行，全部通过原物理梯度门限及独立AD/解析导数检查。编译后有扰动8.79–9.07秒、无扰动2.22–2.24秒；比之前原dense LM更快，仍慢于L-BFGS成功样本。不能从这些小样本推导普遍稳定性保证。

后续[性能拆解与优化实验](gauss_newton_performance_results.md)保留上述初版记录，另以可选fast组合降至约2.71秒/1.16秒；同一最终物理门不变。该报告还保留了收益有限的compiled对角实验。

## 论文与SciPy接口

[论文§1.4式(18)](https://arxiv.org/html/2510.07382v1#S1.SS4)称其为Newton iteration；按它采用的曲率矩阵，严格说属于正则化Gauss–Newton：

\[
(J^TWJ+Q)\Delta x=-J^TW(f(x)-d)-Qx.
\]

它没有完整Newton中的残差加权模型二阶导数项。论文描述每轮累加完整步、实际使用20轮，外层再优化正则化强度。本实验使用既有图像Mock、固定强度和原初值；比较的是求解方法，不是复现其VLBI数据、零初值或外层evidence搜索。

SciPy没有一个叫gauss_newton的直接入口。这里使用官方现成的、带信赖域保障的GN方法：

~~~python
result = scipy.optimize.least_squares(
    residual, z0,
    jac=jax_linear_operator,
    method="trf", tr_solver="lsmr",
    loss="linear", x_scale=1.0,
    ftol=None, xtol=None, gtol=1e-12,
    tr_options={"atol": 1e-10, "btol": 1e-10,
                "maxiter": 200, "regularize": True},
    callback=original_gradient_gate,
)
~~~

TRF通过实际/预测下降调节信赖域，LSMR近似求GN方向。因此它属于**带步长保障的高斯–牛顿方案**，不是论文中无条件接受完整GN步的逐字复刻。[SciPy官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)、[本环境版本的官方源码](https://github.com/scipy/scipy/blob/v1.17.1/scipy/optimize/_lsq/trf.py)。

## JAX提供什么，目标有没有变化

最小二乘接口需要残差及其Jacobian，只有scalar penalty的梯度不足以构造GN曲率。保留原完整精度Q、累积物理状态x=(source,potential)，以及原初始白化：

\[
Q=L_QL_Q^T,\quad H_0=J_{w,0}^TJ_{w,0}+Q=L_0L_0^T,
\quad x=x_0+L_0^{-T}z,
\]
\[
r(z)=\begin{bmatrix}(f(x)-d)/\sigma\\L_Q^Tx\end{bmatrix},
\qquad \tfrac12\|r(z)\|^2=F(x).
\]

物理Q包含相同源平滑、势四阶差分和所有势节点的零阶幅度正则化。没有新增物理ridge、质量片重标定或非负约束。SciPy的regularize=True只影响临时线性子问题，不改变上述F。

`jac(z)`返回[LinearOperator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html)，JAX [JVP](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html)计算Jv，[VJP](https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html)计算Jᵀu，另提供两列批量matmat。每个operator保存自己的z副本，编译函数身份保持不变，z和分解矩阵作为动态参数。残差和动作返回可写NumPy数组，支持一维和列向量输入。

这样避免了每轮形成25373×3357、约681MB的完整Jacobian及其CPU dense QR。初始白化H0、Q的Cholesky仍是dense；LSMR/TRF的控制与部分向量计算仍在CPU。这里的“无矩阵”仅指每轮当前Jacobian的计算接口，不能称整个流程完全无矩阵或完全在GPU。

每次fit检查残差平方和与原标量F相等、方向导数链式法则和JVP/VJP的adjoint恒等式。终点在计时内重算当前GN尺度梯度，返回后再用独立AD/原解析梯度双重检查。

## 统一停止与验收

继续使用原当前物理尺度梯度：

\[
\gamma(x)=\max_i\frac{|\nabla F(x)_i|}{\sqrt{(J_w^TJ_w+Q)_{ii}}},
\qquad\gamma\le a_{tol}+10^{-8}\gamma(x_0).
\]

subhalo的atol=5e-4、control=3e-5，未为新方法放宽。默认每个外层callback检查原门限，通过后raise StopIteration；保存的停止点还需通过最终AD和解析证书才接受。

主动callback停止在SciPy中是status=-2、success=False，这两个native字段原样保留，accepted_reconstruction表示独立物理验收。预算停止、native失败和物理收敛分别记录。native optimality可能在callback停止时沿用上轮值，不能代替重算证书。ftol/xtol默认关闭，避免仅因目标/步长变化小而提前结束；保留gtol=1e-12作为native条件。

max_seconds默认120秒，是外层callback检查的软求解预算，不是硬进程时限。若出现未验收终点，脚本仍保存诊断并最终返回非零状态。

## 原始200×200 Mock：相同真值节点初值

使用原 [科学配置](../../examples/potential_correction/iterative_scientific_configuration.json)：22016个有效数据，2401源变量、956势变量，共3357变量。源49×49、势间距0.075 arcsec、相同PSF/mask/noise、nsub=2、全部正则化强度不变。宏模型固定真值，源从原truth-node值开始，势从零开始；数据与保存dense LM baseline逐项核对。

每case首调用及2次保留JIT的重复，每次重新构造数值H0和两个Cholesky，不复用数值白化缓存。

| 案例 | 通过/尝试 | 首调用/秒 | 两次warm/秒 | 外层迭代 | 最终梯度 / 门限 |
| --- | ---: | ---: | --- | --- | --- |
| subhalo | **3/3** | 14.885 | 8.793、9.066 | 23、23、24 | 3.6224e-4–4.9046e-4 / 5.0098e-4 |
| control | **3/3** | 5.176 | 2.215、2.238 | 9、9、9 | 2.9961e-5 / 3.0055e-5 |

subhalo的F=11734.8265192–11734.8265303，原dense LM为11734.8265138；模型图像相对原LM差1.38e-5–1.99e-5倍noise RMS。恢复的离散kappa峰约0.303596，位置(0.75,0.075) arcsec，与原LM峰0.303586一致。离散注入真值峰约0.631435，因此原正则化下的峰值压低仍然存在，不能声称恢复了无偏质量或检测显著性。

control的F=11404.443655827，较原LM高约1.70e-7；图像差6.94e-7倍noise RMS，all-core kappa RMS=0.000532932，与原control基本一致。

![有扰动：首调用，通过原梯度验收](gauss_newton_figures/truth_subhalo_perturbation.png)

![无扰动：首调用，通过原梯度验收](gauss_newton_figures/truth_control_perturbation.png)

## 从观测构造初始源

另外在相同200配置、p=0和固定宏模型下，先只用观测做source-only MAP初值，再启动新GN；没有读取source truth。

| 案例 | 验收 | F | 当前梯度 / 门限 | 外层迭代 |
| --- | --- | ---: | --- | ---: |
| subhalo | **通过** | 11734.826516403 | 3.3658e-4 / 5.0017e-4 | 24 |
| control | **通过** | 11404.443655560 | 1.4388e-5 / 3.0000e-5 | 6 |

两例独立AD/解析证书均通过，模型图像与原LM差分别为1.20e-5和9.30e-7倍noise RMS。这支持预条件和求解不依赖已知源真值；宏模型仍固定真值，不能扩展成宏模型未知时的验证。该批用于正确性检查，图像导出曾与准备阶段重叠，不用于下面速度比较。

## 与之前方法的速度关系

以下沿用原200配置、truth-node初值；原LM/L-BFGS来自既有[计时实验](timing_results.md)，没有重新运行或删除其失败记录。

| 方法 | 有扰动warm/秒 | 无扰动warm/秒 | warm验收/尝试 |
| --- | --- | --- | --- |
| 原dense LM | 33.791–33.827 | 13.908–13.937 | 3/3、3/3 |
| 本次SciPy TRF/LSMR GN | 8.793–9.066 | 2.215–2.238 | 2/2、2/2 |
| 原AD + L-BFGS成功样本 | 1.072–1.121 | 0.701–0.718 | **2/3**、3/3 |

本次是不同批次、2次warm的小规模计时，不能据此宣称高可靠性或精确通用加速比。记录支持当前GN比原dense LM快、比L-BFGS成功样本慢。之前L-BFGS有一个warm记录未过原门限，未混进成功样本时间。

与[lmfit实验](lmfit_results.md)相比，完整200对照由约421.5秒降至本次首调用5.18秒/warm约2.23秒；lmfit有扰动则在680.5秒预算退出、未通过门，不能拿其失败时间计算成功速度比。优势来自具体后端/线性求解方式，不是“所有GN必定比所有LM快”。

本次warm subhalo的8.793秒中：初始metric约0.458秒、两项分解约0.071秒、SciPy API约8.100秒；API里JVP/VJP/批量动作约4.67秒、逐轮物理检查约3.28秒。control的2.215秒中逐轮检查约1.27秒。验收成本已计入，没有只报告优化器内核时间；后续若优化检查调度，应另做研究并继续维持同一最终门限。

计时从已准备problem开始，到原始场产品完成，包含上述准备、初始目标/导数检查、CPU/GPU传输、全部外层/内层工作和原梯度验收。排除Mock/几何生成、最终文件导出、绘图和返回后的额外独立解析审计；API内进度打印计入。与历史benchmark相比还多初始adjoint核对、prior Cholesky和逐轮callback验收等工作，表格给出实际实现的完整恢复成本，不是相同内核微基准。

## 验证与复现

小型非线性耦合例（37变量）3/3通过，F=0.446379594795874；小型透镜节点例（122变量）3/3通过，F=0.0460958021480613。主报告的全部8个完整Mock终点均由独立AD/解析门再次验证，代码快照、环境、输入配置和hash随输出保存。

另外验证两条失败路径：[max_nfev=2](gauss_newton_eval_budget/summary.json)为native status=0，[极小wall预算](gauss_newton_wall_budget/summary.json)为status=-2、callback_wall_budget；两者均保存真实终点与history、accepted=False、独立梯度不通过，并按设计返回退出码1。没有把预算退出归为收敛。所有16条记录的代码快照hash、残差目标身份、history终点和状态均完成自动核对；脚本与绘图通过语法和格式检查。

~~~bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu

# 原200配置；首调用加两次warm；使用新的输出目录
python -u workspace/pt_new_algorithm/run_gauss_newton.py --repeats 2 \
  --output workspace/pt_new_algorithm/gauss_newton_recheck

# 仅从观测构造源初值
python -u workspace/pt_new_algorithm/run_gauss_newton.py \
  --initialization source_only \
  --output workspace/pt_new_algorithm/gauss_newton_source_recheck

# 秒级小型透镜例
python -u workspace/pt_new_algorithm/run_gauss_newton.py --fixture nodal \
  --repeats 2 --output workspace/pt_new_algorithm/gauss_newton_nodal_recheck
~~~

原始记录：[full truth](gauss_newton_full_truth/summary.json)、[source-only](gauss_newton_full_source_only/summary.json)、[coupled](gauss_newton_coupled/summary.json)、[nodal](gauss_newton_nodal/summary.json)。独立主源/实现/结果核对见 [gauss_newton_review.md](gauss_newton_review.md)。所有成功/失败均保留在各自新目录，不覆盖旧方法的记录。
