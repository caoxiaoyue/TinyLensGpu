# Potential correction：自动微分与 L-BFGS 可行性核对

2026-11-03。本文只核对数学问题和第三方接口；实测结果见同目录的实验记录。所有结论限定于**固定宏模型、固定源/势网格、固定正则化强度的有符号源—势联合优化**。

## 核对结论

这个方向成立：把当前非线性正则化目标交给 `jax.value_and_grad` 和成熟 L-BFGS 实现，可以省去手写解析 Jacobian 与 LM 控制。两者求解的是同一优化问题，但数值算法不同；“更稳定、更快”仍需对照实验。

令 \(x=(s,p)\)，\(p=\delta\psi\)，\(f(x)=B L(p)s\)，\(W=\operatorname{diag}(\sigma^{-2})\)。当前目标恰为

\[
F(s,p)=\frac12\|\sigma^{-1}(d-f(s,p))\|^2
+\frac12\lambda_s s^T R_s s
+\frac12\lambda_p\|D_4p\|^2
+\frac12\lambda_a\|p\|^2.
\]

其中源正则化保持当前物理像素尺度，势的完整四阶差分使用当前像素单位；\(\lambda_a\) 是所有势节点上的独立物理幅度先验，不能在实验中当作纯数值 damping 删除。代码依据：[_evaluate](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py:283)、[prior_matvec](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/_potential_correction_common.py:133)、[势先验](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/utils/inversion/potential_regularization.py:66)。

写 \(Q=\operatorname{diag}(\lambda_sR_s,\lambda_pD_4^TD_4+\lambda_aI)\)，\(J=\partial f/\partial x\)，则

\[
\nabla F=-J^TW(d-f)+Qx,\qquad H_{\rm GN}=J^TWJ+Q.
\]

现有代码解析计算的是所选双线性分支的精确梯度，以及 **Gauss–Newton 曲率**；后者通常并非完整 Hessian。完整 Hessian 还含 \(-\sum_i[W(d-f)]_i\nabla^2 f_i\)。LM 用 \(H_{\rm GN}+\mu\operatorname{diag}(H_{\rm GN})\) 求步长，L-BFGS 用梯度差更新有限记忆逆 Hessian 近似。自动微分负责求导，第三方优化器负责迭代，这两个选择彼此独立。代码依据：[解析 Jacobian](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py:262)、[当前 curvature 构建](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py:112)、[LM 增量](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py:69)。

局部参考实现也直接声明并求解相同结构的 penalized objective，但其默认 \(\mu I\) damping 与当前 TinyLensGpu 的对角尺度 damping 不同：[参考 iterative.py](/home/xycao/workspace/PT_dev/lensing_potential_correction/potential_correction/iterative.py:259)。这进一步说明不能把不同求解器的步长、迭代次数直接视作等价。

## 实验必须保持的条件

- **优化整个非线性正向模型**：每次由当前 \(p\) 更新射线和源插值，再由当前 \(s\) 生成图像。只优化首次线性化的矩阵，测到的是单步线性问题，不是本提议。[forward](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py:253) 已可用于构建标量目标。
- **固定同一问题**：LM 与 L-BFGS 使用同一数据、mask、PSF、初值、网格、所有正则化项和精度。源与势均允许有符号系数；加入非负约束会改变比较对象。
- **分支可微性**：双线性插值在单元内部可微，内部格点处连续但梯度可跳变；当前实现把源网格外亮度设为零，若边缘源值非零，穿越外边界还可能令函数跳变。AD 返回执行分支的导数，不能消除这些问题。[映射与分支梯度](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/utils/lensing/mapping.py:40)。先在避开分支边界的点验证 AD 梯度与解析梯度/有限差分，再单独记录跨单元优化的行为。
- **参数尺度**：源亮度与势系数量级不同，势四阶差分也会造成较差条件数。可用固定可逆变换 \(x=x_0+Tz\)，优化 \(F(x_0+Tz)\)；这只改变坐标、不改变最优问题。对角尺度成本低；先验或初始曲率的完整 Cholesky 白化适合小型对照，但具有平方存储/立方分解成本，不能据此宣称大规模方案已成立。AD 梯度在新坐标应为 \(T^T\nabla_xF\)。
- **收敛验收**：同时报告目标函数、\(\chi^2\)、两类正则化 penalty、梯度范数、迭代/函数评估次数和终止原因；只看第三方 `success` 或目标下降不足以确认达到近似驻点。单个 mock 也不能证明全局最优或真实子结构恢复。

以上是根据代码得出的实验要求，不是已经测得的性能结论。

## 可直接使用的第三方接口

最小 CPU 调度/GPU 计算版本：

```python
value_grad = jax.jit(jax.value_and_grad(objective_in_scaled_coordinates))

def scipy_fun(z):
    value, grad = value_grad(jnp.asarray(z, dtype=jnp.float64))
    return float(value), np.asarray(grad, dtype=np.float64)

result = scipy.optimize.minimize(
    scipy_fun, z0, method="L-BFGS-B", jac=True,
    options={"maxiter": 2000, "maxls": 50, "ftol": 1e-13, "gtol": 1e-7},
)
```

不设置 bounds 时保留当前无约束有符号问题；`jac=True` 明确传入 AD 梯度。每次把标量和梯度传回 NumPy 都发生设备同步/传输，所以此版本适合验证可行性，不足以代表纯 GPU 优化器的性能。上述容差只是初始实验值，应与实际尺度和精度一起记录。[JAX value_and_grad 接口](https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html)、[SciPy minimize 的 jac 参数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)、[L-BFGS-B 的 ftol/gtol/maxls 说明](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)。

纯 JAX 备选是 `optax.lbfgs`，默认使用 line search；官方示例将目标值、梯度及 `value_fn` 交给 `opt.update`，循环可以编译。在相同目标和尺度下复核结果后，才比较设备性能。[Optax lbfgs API](https://optax.readthedocs.io/en/latest/api/generated/optax.lbfgs.html)、[官方示例](https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html)。当前环境虽有 JAXopt 0.8.5，其官方仓库已声明不再维护和开发，并提示部分功能已迁入 Optax，因此不建议把它作为后续整合的默认依赖。[JAXopt 状态](https://github.com/google/jaxopt#status)。

## 两个需要分开的后续问题

**固定强度 MAP 与 evidence**：只替换 \((s,p)\) 的求解器可以保持本次目标。正则化强度若也自由最小化这个裸 penalty，会产生偏向减小正则化的退化，不等价于 evidence 优化。当前 evidence 还含先验与 posterior log-determinant，且明确采用未阻尼 GN 曲率的局部高斯近似；L-BFGS 记忆中的逆 Hessian 近似不能直接替代其 log-determinant。[当前 evidence 公式](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py:624)。Powell 等 2025 年预印本 §1.4、式 (13)–(18) 同样区分内层源—势迭代与外层正则化强度搜索：[论文](https://arxiv.org/html/2510.07382v1#S1.SS4)；本地阅读版本是 [2510.07382v1.pdf](/home/xycao/workspace/PT_dev/paper_pdf/2510.07382v1.pdf)。技能中另一个本地引用 `staf2253.pdf` 不存在，本核对未声称验证该文。

**联合优化与 variable projection**：本次最直接比较联合变量 \((s,p)\)。另一个有根据的方向是对每个 \(p\) 精确求解有符号线性源问题

\[
s_*(p)=[A(p)^TWA(p)+\lambda_sR_s]^{-1}A(p)^TWd,
\quad \widetilde F(p)=F(s_*(p),p),\quad A(p)=BL(p).
\]

若内层系统可逆且源求解精确，消去源不改变联合问题的全局最小值，但减少外层维度，并增加每步的源线性求解。可微线性求解或 envelope theorem 能给出外层导数；源正则化、边界条件与内层精度都必须保持。它值得后续单独验证，不应与本次“只换优化器”的收益混在一起。

因此，第一步宜用现有小型 nodal mock、float64、固定强度，完成 AD/解析梯度一致性检查，再对照 LM 与缩放后 L-BFGS 的目标、重建及终止状态。通过后再扩展至带 PSF/噪声、更细网格和独立生成的图像。
