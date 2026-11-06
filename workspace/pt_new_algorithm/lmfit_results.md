# lmfit Levenberg–Marquardt：独立 potential-correction 实验

2026-11-03。新文件为 [run_lmfit.py](run_lmfit.py)。已在 tinylens_gpu 环境安装 **lmfit 1.3.4**（新增 asteval 1.0.10、uncertainties 3.2.3；原 JAX 0.7.1、NumPy 2.4.6、SciPy 1.17.1 未改变）。没有修改核心库的求解器或依赖声明。

**结论：相同目标可以用JAX导数和lmfit LM求解，但这版实现没有显示速度或收敛优势。** 小例通过；原200×200对照通过、用时421.5秒，有扰动案例在600秒软预算后仍未过梯度门、实际用时680.5秒。主要耗时为CPU上的MINPACK dense求解。暂不建议用该实现替换原GPU LM或之前的L-BFGS原型。

## 计算的是同一个目标

保留与 L-BFGS-B 实验相同的完整非线性 forward、初始白化、signed 源/势、网格、PSF、mask 和三个物理正则化强度。令 Q 为原完整 prior precision，x 为累积的物理源和势，而非增量：

\[
Q=L_Q L_Q^T,\qquad
r(x)=\begin{bmatrix}(f(x)-d)/\sigma\\L_Q^Tx\end{bmatrix},
\qquad \tfrac12 r(x)^Tr(x)=F(x).
\]

新优化器调用为：

~~~python
minner = lmfit.Minimizer(residual, params, iter_cb=callback, scale_covar=False)
result = minner.leastsq(
    Dfun=jax_jacobian, col_deriv=True,
    ftol=1e-13, xtol=1e-13, gtol=1e-13,
)
~~~

residual 和 Jacobian 在 GPU 上由 JAX 计算；真正的 LM 更新由 lmfit 调用 SciPy/MINPACK 完成。Jacobian 按16列分块自动微分，回调返回 n_variables × n_residuals 的可写 float64 数组。没有调用旧手写 LM，也没有让 lmfit 做有限差分。

每次运行验证 0.5×残差平方和与原目标一致，并独立核对方向导数；终点仍检查原当前 GN 对角尺度梯度。原解析梯度只用于独立审计，不供新优化器使用。

lmfit 的 chisqr 在这里等于 **2F**，包含正则化残差，不是只有图像的 chi²。其 redchi/AIC/BIC 和内部坐标误差条不作为本物理模型的统计结论。

## 原生停止与外部梯度停止分别记录

默认保留 lmfit 自己的停止方式、真实 covariance 和 uncertainty/correlation 后处理，单独计量这些开销。scale_covar=False 只禁止额外的 reduced-chi² 缩放；lmfit 的 leastsq 即使设置 calc_covar=False 也不会关闭这些操作，因而本实验没有假装该选项可以关闭它们。

可选 **--stop-on-gradient** 使用 lmfit 官方 iter_cb，在原物理梯度门限通过时主动结束。LM 步长与阻尼仍全部由 lmfit 决定。该模式的 native aborted=True / success=False 是主动回调的结果，不会篡改；外部是否接受取决于保存的停止原因，以及最后重新计算的 AD 和解析梯度证书。它不是“MINPACK原生收敛”，也不包含被回调中止所跳过的 covariance 后处理。

当前 lmfit/SciPy 组合在中止后返回的参数，曾与正常回调中的参数不一致；底层内存生命周期尚未进一步定位。首次停止点因此显式复制，最终解码和验收使用该副本。内部 max_nfev 中止则使用最近正常回调的副本作为失败诊断；它不变成成功。原 native flags 与副本来源一起保存。

max_seconds 是残差回调检查的求解预算，不是硬进程超时：不能中断正在进行的一次 CPU QR，也不限制随后 native covariance 后处理。因此实际耗时可能超过预算，必须查看记录的真实 wall time。

## 小规模正确性验证

| 案例 | 变量数 | 停止方式 | 验收 | 最终 F | 当前 scaled gradient | 编译后总时间 |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| coupled | 37 | native lmfit | 3/3通过 | 0.44637959476212 | 约6.19e-9 | 0.030–0.031秒 |
| nodal | 122 | native lmfit | 3/3通过 | 0.046095802148056 | 约1e-13（重复） | 0.040–0.041秒 |
| nodal | 122 | 外部原梯度门限 | 3/3通过 | 0.0460958021481 | 约2.30e-8 | 0.035–0.036秒 |

均与原 LM 目标一致。另有两种预算中止 smoke，验证不提供 native chisqr 时仍能保留原始诊断，并且不会把预算结束误标为成功。

最终代码另通过 [max_nfev=4中止回归](lmfit_eval_abort_smoke/summary.json)：native最终参数确实与保存的正常回调副本不同，但导出的F=0.480381374112与最后正常history条目一致，endpoint_source明确为owned_last_completed_callback_before_native_abort，accepted=False，缺失chisqr按null保存。这验证了内部评估预算先于用户回调中止时的保护路径。

## 原有64×64科学校准配置：原生停止

使用仓库已有 [64配置](../../.scratch/potential-correction/research/iterative-scientific/calibration64-configuration.json) 和保存的同配置 dense LM 场。实际为2252个有效像素、2401个源变量、936个势变量，共3337变量；PSF5×5、dpix=.0375、生成/反演nsub=2，source49×49及物理先验保持该配置原值。它与200×200的采样不同，结果不混用。

| 案例 | lmfit native success | 原物理门限验收 | 新 F | 原 LM 的 F | 梯度 / 门限 | 完整调用 |
| --- | --- | --- | ---: | ---: | --- | ---: |
| subhalo | True，ier=1 | **未通过** | 1661.127368388 | 1661.127375772 | 2.2951e-4 / 3.0130e-5 | 534.067秒 |
| control | True | **通过** | 1556.968863140 | 1556.968863224 | 7.5851e-7 / 3.0021e-5 | 265.021秒 |

subhalo 的目标和恢复场接近原解，峰值0.2176603，原64基线0.2176568，位置均(0.75,.075) arcsec。但它没有通过既定分支梯度门限，不能因为 native success 或图像接近而算验收成功，也不能据此断定该点一定不是非光滑局部最优。

时间分解：

| 项目 | subhalo | control |
| --- | ---: | ---: |
| lmfit API 总时间 | 528.687秒 | 262.800秒 |
| 其中 JAX Jacobian 总时间 | 17.794秒（29次） | 9.299秒（15次） |
| covariance 坐标变换 | 0.018秒 | 0.018秒 |
| uncertainty/correlation 后处理 | 4.304秒 | 4.100秒 |

主要耗时发生在 CPU 内部求解，而不是 JAX 导数或 lmfit 的相关系数后处理。该套 dense MINPACK QR 主要占用一个 CPU 核；设置16个 BLAS线程并不使这些核心循环自动并行或转到GPU。

![64有扰动：未过物理梯度门限的终点诊断](lmfit_calibration64_figures/truth_subhalo_perturbation.png)

![64控制：通过验收的恢复对照](lmfit_calibration64_figures/truth_control_perturbation.png)

## 原始200×200配置

使用原3357变量、22016有效像素配置，augmented Jacobian为3357×25373，单份约681MB；64配置单份约149MB。两者均按同样物理参数定义，单次大规模 Jacobian 生成约0.8秒，但200配置的相邻 Jacobian 调用间 CPU 求解约80多秒。

第一次 control 原生停止探针触发600秒回调预算后，旧快照的诊断保存曾假设 native chisqr 总存在，因而保存异常。其 [failure manifest](lmfit_full_control/probe_failure.json) 只保留耗时/进度证据，没有终点或梯度证书，不能判为物理已收敛或未收敛。当前文件已修复；该早期失败不覆盖、不伪造恢复数据。

原200配置另跑了匹配物理梯度停止的变体（--stop-on-gradient、每例600秒软预算）。这不放宽物理门限，且与上面的原生停止分开报告：

| 案例 | 原物理门限验收 | 终止原因 | F | 梯度 / 门限 | 单次 fit 时间 |
| --- | --- | --- | ---: | --- | ---: |
| subhalo | **未通过** | 回调预算结束 | 11739.649030480 | 0.528642 / 0.000500976 | 680.528秒 |
| control | **通过** | 回调达到原梯度门限 | 11404.443655657 | 1.61244e-5 / 3.00554e-5 | 421.528秒 |

subhalo 从 F=88493.902750133 降到上述值，仍比原 LM 的11734.826513786高4.822517。预算后保存的点来自首次停止回调的独立副本，AD和解析梯度均为0.528642，不能作为成功恢复。lmfit API用时674.803秒，其中9次完整AD Jacobian共7.308秒，回调物理梯度检查共2.153秒；剩余主要是CPU求解。单次CPU求解跨过600秒预算，因此实际结束时间更晚。这一例没有执行native covariance/uncertainty后处理，其慢速不能归于这些后处理。

control 的AD和解析梯度分别为1.61243627e-5与1.61243628e-5，均通过原门限；目标与原LM相差约-5.1e-10，模型图像RMS差为1.85e-6倍noise，kappa最大绝对差3.93e-7。API用时418.871秒，其中6次AD Jacobian共5.025秒、外部梯度检查1.662秒。它因官方回调主动结束而native success=False、aborted=True；“通过”专指保存点经独立验收，并不宣称MINPACK原生停止成功。两例的停止副本均与native最终参数一致，但仍使用已保护的副本导出。

计时范围从已装配problem开始，包含初始自动微分白化/先验分解、参数准备、完整API调用、终点当前梯度检查及原始场产品；不含Mock生成、后续额外解析审计、比较图和最终结果导出。Jacobian回调中的checkpoint写入和进度输出计入API时间。这里只运行每例一次，不给出warm中位数，也不把未通过验收的时间计算为成功速度比。

此前相同200配置的[计时实验](timing_results.md)中，原dense LM首次约38.25秒/18.44秒，L-BFGS首次约5.13秒/4.69秒（subhalo/control）。范围有后处理和checkpoint等细节差异，但当前lmfit的CPU求解已达数百秒，明显没有表现出速度优势；不能据此将差异泛化成所有LM实现的算法速度差。

![200有扰动：预算终点，未过原梯度门限](lmfit_full_gradient_figures/truth_subhalo_perturbation.png)

![200控制：通过原梯度门限](lmfit_full_gradient_figures/truth_control_perturbation.png)

## 运行与数据

~~~bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python -m pip install lmfit==1.3.4  # 本次环境已安装；其他环境复现时需要

# 原始200配置，保留 lmfit 原生停止与后处理
python -u workspace/pt_new_algorithm/run_lmfit.py \
  --output workspace/pt_new_algorithm/lmfit_native_recheck

# 相同原模型，按原物理梯度门限回调停止
python -u workspace/pt_new_algorithm/run_lmfit.py --stop-on-gradient \
  --max-seconds 600 --max-nfev 80 \
  --output workspace/pt_new_algorithm/lmfit_gradient_recheck

# 秒级小例验证
python -u workspace/pt_new_algorithm/run_lmfit.py --fixture nodal \
  --stop-on-gradient --repeats 2 \
  --output workspace/pt_new_algorithm/lmfit_nodal_recheck
~~~

64复现需显式使用 --configuration 指向上面的64配置、--baseline 指向 .scratch/potential-correction/output/iterative-scientific/calibration64。所有复现使用新输出目录。

原始数据：[coupled](lmfit_coupled/summary.json)、[nodal](lmfit_nodal/summary.json)、[外部门限smoke](lmfit_external_smoke_v2/summary.json)、[64 native](lmfit_calibration64/summary.json)、[200原梯度回调](lmfit_full_gradient/summary.json)。每组保存环境、输入配置、代码快照和hash、逐回调历史、停止原因、独立审计、原始场和CPU/GPU分段时间。

独立的官方接口/数学/实现核对见 [lmfit_review.md](lmfit_review.md)。参考 [lmfit fitting文档](https://lmfit.github.io/lmfit-py/fitting.html) 与 [SciPy leastsq文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html)。

没有拟合宏模型、调节正则化强度、添加硬质量片约束或计算新 evidence。lmfit统计量、局部曲率和控制图像的残余起伏不作为检测显著性或扰动质量置信区间。
