# 用 JAX 自动微分、白化和 L-BFGS 做 potential correction

记录日期：2026-11-06。

本教程从接近 Python 的伪代码出发，解释如何联合优化源亮度与势修正。
核心是：先构造一个固定白化变换，再让 SciPy 的 L-BFGS-B 在新坐标中优化，
由 JAX 计算完整非线性目标及其梯度。

教程和改进设想先保存在 `workspace/iteration/`，供本轮实验理解与后续筛选使用。
下面的稠密代码用于说明流程，不是本目录真实数据实验的直接运行入口。

## 1. 原来要优化什么

把源亮度节点与势修正节点拼成联合变量：

\[
x=\begin{pmatrix}s\\p\end{pmatrix},
\qquad x\in\mathbb R^{N_s+N_p}.
\]

- \(s\)：全部源亮度节点。
- \(p\)：相对于固定平滑宏模型的累计势修正节点。
- 平滑宏模型、数据、噪声、PSF、mask 和网格在一次内层优化中固定。

势修正改变偏折，因此改变图像像素在源平面的取样位置：

\[
\beta(p)=\beta_{\rm macro}-A_\alpha p.
\]

这里 \(A_\alpha\) 表示势节点到图像取样位置处偏折的固定线性算子。
源亮度在移动后的坐标处插值，再经过子像素积分、PSF 和观测像素选择，得到 \(f(s,p)\)。
固定 \(p\) 时，图像对源亮度是线性的；联合改变 \(s,p\) 时，完整问题是非线性的。

```text
源节点 s ───────────────────────┐
                              ↓
势节点 p → 修正偏折 → 源坐标 β → 源亮度插值 → 子像素积分/PSF → 预测图像 f(x)
```

在独立高斯噪声下，定义已经除以噪声的残差：

\[
r(x)=\frac{f(x)-d}{\sigma}.
\]

固定正则化强度时，优化目标是负对数后验中与 \(x\) 有关的部分：

\[
F(x)=\frac12r(x)^\mathsf Tr(x)+\frac12x^\mathsf TQx,
\]

\[
Q=\begin{pmatrix}
\lambda_sR_s&0\\
0&\lambda_4D_4^\mathsf TD_4+\lambda_0I
\end{pmatrix}.
\]

\(R_s\) 是固定权重下的源正则化矩阵，\(D_4\) 汇集势的四阶差分。
本教程假设 \(Q\) 正定，使先验与白化均有定义；它是事先指定的物理目标的一部分。
这里沿用允许有符号的源和势系数，不设置非负约束。

## 2. JAX、SciPy 和 NumPy 各做什么

| 部件 | 职责 | 典型调用 |
| --- | --- | --- |
| JAX | 前向计算、自动微分、编译数值计算 | `jnp`、`jax.jacrev`、`jax.value_and_grad`、`jax.jit` |
| SciPy | L-BFGS-B 搜索方向、线搜索、有限历史和停止条件 | `scipy.optimize.minimize` |
| NumPy | 将标量和梯度转换为 SciPy 接受的宿主端数据 | `float(value)`、`np.asarray(gradient)` |

`forward(x)` 内部需要使用 JAX 支持的可微操作。转回 NumPy 的工作放在 SciPy 接口处，
不能在待求导的前向路径中随意把 JAX tracer 转为 NumPy 数组。

如果 JAX 使用 GPU，前向和梯度可在 GPU 上计算；SciPy 的优化调度仍在 CPU 上。
每次将目标值和梯度交回 SciPy 都涉及同步或设备传输。

## 3. 先对白化坐标建立直觉

源亮度和势系数可能具有很不相同的尺度，源与势还可以部分补偿彼此对图像的影响。
在原坐标中，局部目标可能是一个狭长、倾斜的谷，给搜索步长和有限历史曲率估计带来困难。

在参考状态 \(x_{\rm ref}\) 计算噪声加权 Jacobian：

\[
J_{\rm ref}=\left.\frac{\partial r}{\partial x}\right|_{x_{\rm ref}},
\qquad
H_{\rm ref}=J_{\rm ref}^\mathsf TJ_{\rm ref}+Q.
\]

这是 Gauss–Newton 曲率。它保留源与势的交叉块
\(J_{{\rm ref},s}^\mathsf TJ_{{\rm ref},p}\)，因此完整白化也处理相关方向。
一般非线性问题的真实 Hessian 还含残差乘以前向二阶导数的项；这里没有将它等同于 GN 曲率。

做 Cholesky 分解，并定义固定的坐标变换：

\[
H_{\rm ref}=LL^\mathsf T,
\qquad T=L^{-\mathsf T},
\qquad x=x_{\rm start}+Tz.
\]

其中 \(x_{\rm start}\) 是实际优化起点，可以与 \(x_{\rm ref}\) 不同。
\(z=0\) 恰好对应 \(x=x_{\rm start}\)。白化的二次型关系是：

\[
T^\mathsf TH_{\rm ref}T=I.
\]

这个变换同时包含缩放与相关方向的混合。由于 \(T\) 固定且可逆，优化
\(\widetilde F(z)=F(x_{\rm start}+Tz)\) 与优化原来的 \(F(x)\) 是同一个问题的不同坐标表示。
实际计算使用三角求解应用 \(L^{-\mathsf T}\)，不需要显式求逆。

## 4. 一组接近 Python 的完整伪代码

下面封装一次固定正则化强度下的内层优化。输入中：

- `forward(x)` 返回选中观测像素上的一维预测图像，并在每次调用时按当前源和势重新渲染。
- `data`、`noise` 是同一像素顺序下的一维数组，`noise` 有限且严格为正。
- `x_initial` 是实际联合初值，`Q` 是已组装好的正定联合先验精度矩阵。
- `x_reference` 可省略，此时以实际初值作为曲率参考。

示例选用 Float64，并显式存储小型问题的 Jacobian 和曲率。
优化选项仅用于说明接口；具体容差仍应由目标尺度、精度和验收协议决定。

```python
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from scipy.optimize import minimize

jax.config.update('jax_enable_x64', True)


def fit_with_whitened_lbfgs(
    forward, data, noise, x_initial, Q, x_reference=None
):
    x_start = jnp.asarray(x_initial, dtype=jnp.float64)
    x_ref = (
        x_start if x_reference is None
        else jnp.asarray(x_reference, dtype=jnp.float64)
    )
    data = jnp.asarray(data, dtype=jnp.float64)
    noise = jnp.asarray(noise, dtype=jnp.float64)
    Q = jnp.asarray(Q, dtype=jnp.float64)

    # 1. 原物理坐标中的目标；正则化作用于完整累计状态 x。
    def residual(x):
        return (forward(x) - data) / noise

    def objective_x(x):
        r = residual(x)
        return 0.5 * (r @ r) + 0.5 * (x @ Q @ x)

    # 2. 在参考点准备一次完整 GN 曲率与白化因子。
    # residual 已经除以 noise，这里不再重复乘噪声精度。
    J_ref = jax.jacrev(residual)(x_ref)
    H_ref = J_ref.T @ J_ref + Q
    L = jnp.linalg.cholesky(H_ref)
    if not np.isfinite(np.asarray(L)).all():
        raise FloatingPointError('Reference Cholesky failed')

    # 3. z -> x；L 在本次 minimize 内固定。
    def decode(z):
        delta_x = solve_triangular(L.T, z, lower=False)
        return x_start + delta_x

    # 4. 对复合函数 F(decode(z)) 求值并对 z 求导。
    def objective_z(z):
        return objective_x(decode(z))

    value_grad_z = jax.jit(jax.value_and_grad(objective_z))

    # 5. SciPy 接口：返回 Python 标量和 NumPy 梯度。
    def scipy_fun(z_numpy):
        z = jnp.asarray(z_numpy, dtype=jnp.float64)
        value, gradient_z = value_grad_z(z)
        return float(value), np.asarray(gradient_z, dtype=np.float64)

    # 6. SciPy 管理 L-BFGS 历史和线搜索；初始 z=0。
    result = minimize(
        fun=scipy_fun,
        x0=np.zeros(x_start.size, dtype=np.float64),
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': 3000,
            'maxcor': 30,
            'maxls': 80,
            'ftol': 1e-15,
            'gtol': 1e-8,
        },
    )

    # 7. 解码最终候选，并另外取得原物理坐标的梯度。
    x_best = decode(jnp.asarray(result.x, dtype=jnp.float64))
    value, gradient_x = jax.value_and_grad(objective_x)(x_best)
    return {
        'x': np.asarray(x_best),
        'objective': float(value),
        'physical_gradient': np.asarray(gradient_x),
        'optimizer': result,
    }
```

这段代码中，`jac=True` 表示 `scipy_fun` 已同时返回目标值和梯度。
SciPy 不需要再通过有限差分估计梯度。`maxcor` 控制有限历史长度，
`maxls` 控制每次迭代允许的线搜索步数；一次接受的迭代可能评估多个试探点。

`jax.jit` 编译目标值与梯度的数值计算，第一次调用包含编译成本。
它不会把外部 SciPy 的整个优化循环自动编译为 GPU 程序。

## 5. 为什么梯度自然变成了对 z 的梯度

实际自动微分的计算路径是：

```text
z → decode(z) → x → forward(x) → 残差和正则化 → F(x)
```

所以 `jax.value_and_grad(objective_z)` 返回：

\[
\nabla_z\widetilde F(z)
=T^\mathsf T\nabla_xF(x)
=L^{-1}\nabla_xF(x).
\]

三角求解也包含在 JAX 的求导路径内，无需另外手写这个梯度变换。
如果只计算 `objective_x` 的梯度后原样交给优化 \(z\) 的 SciPy，目标与梯度就不匹配。

白化只改变搜索坐标，前向始终使用当前物理解 \(x\)，先验也始终惩罚累计状态 \(x\)。
它没有把目标改成只惩罚增量 \(Tz\)，也没有把非线性图像固定为参考点的线性展开。

## 6. L-BFGS 的优势体现在哪里

完整 Jacobian 和白化因子在准备阶段构造一次。随后每个试探点主要需要目标值与梯度，
无须在每步显式形成并分解完整 Hessian。

L-BFGS 根据最近若干步的位移和梯度差，近似逆 Hessian 对梯度的作用。
若保留 \(m\) 组历史、共有 \(n\) 个参数，历史存储量为 \(O(mn)\)，
通常远小于显式存储完整逆 Hessian 的 \(O(n^2)\)。这并不消除上面稠密白化的准备成本。

SciPy 内部过程可以粗略理解为下列伪代码；实际还包括曲率对筛选、线搜索与退出处理：

```text
先计算当前位置的目标值和梯度
循环：
    用有限历史和当前梯度，计算搜索方向
    沿搜索方向尝试步长，调用 scipy_fun 检查目标值和梯度
    接受新位置
    记录本步位移与梯度变化，更新有限历史
    检查停止条件或计算预算
```

固定的是白化变换；L-BFGS 的历史曲率信息仍随迭代更新。

## 7. 对应到本目录实际实现

| 教程中的部分 | 本目录的对应实现 |
| --- | --- |
| `forward(x)`、原目标 | [lbfgs.py](lbfgs.py) 中的 `problem.forward` 与 `objective` |
| 参考 Jacobian | `EvidenceEngine.__init__`、`jacobian`，使用分批 VJP 构造 |
| 白化准备与 `decode` | `prepare_coordinates`、`low_rank_whiten`、`decode` |
| JAX 目标值与梯度 | `white_value_grad` |
| SciPy 适配和内层优化 | `fit` 中的 `fun`、`minimize` |
| 输入、外层搜索和记录 | [pt.py](pt.py) 及 C1 系列运行脚本 |

当前 J0946 配置有 10000 个源节点、1134 个势节点和 2293 个有效数据像素。
实际实现利用先验分块分解及数据空间低秩运算应用白化，避免形成完整的
11134×11134 联合曲率；源和势各自的先验因子仍需要准备。

数学上，若 \(Q=AA^\mathsf T\)、\(V=A^{-1}J_{\rm ref}^\mathsf T\)，则可使用：

\[
T=A^{-\mathsf T}(I+VV^\mathsf T)^{-1/2}.
\]

它同样满足 \(T^\mathsf TH_{\rm ref}T=I\)，但不必与教程的三角白化矩阵逐元素相同。
低秩实现通过数据空间的 \(I+V^\mathsf TV\) 分解应用这个变换，仍保留源–势耦合。
从优化流程看，主要是把教程中的 `decode` 换成这个白化作用。

这里还要区分两个状态：

- 参考点用于构造曲率；原 J0946 使用保存的 M3 源亮度与零势修正。
- 实际起点用于启动优化；原 `pt.py` 默认按候选强度求 source-only MAP，
  C1 对照及其后续搜索使用冻结的共同初值，见 [C1 对照](cubic_source_results.md)
  和 [三强度搜索](run_cubic_evidence_joint.py)。

C1 实验继续共享旧 hard-bilinear M3 的参考 Jacobian。
因此白化恒等式描述所采用参考模型的 GN 二次型，不表示实际 C1 模型在起点或终点的曲率恰好为单位阵。

## 8. 收敛检查、曲率变化和后续改进

随着 \(x\) 更新，当前曲率一般会改变。对固定变换有：

\[
T^\mathsf TH_{\rm GN}(x)T
=I+T^\mathsf T[H_{\rm GN}(x)-H_{\rm ref}]T.
\]

所以固定白化可以改善基准尺度与相关性，却不保证后续曲率始终为单位阵。
L-BFGS 的有限历史继续适应局部变化；若参考已经明显失效，可以研究分段重白化。
该方向已记在 [未来改进 TODO](future_improvements.md)，暂未实现或验证。

一次 `minimize` 返回后，应检查 `result.success`、`result.message`、有限性和原物理目标下的驻点诊断。
SciPy 可能因目标值变化很小而停止，仅有 `success=True` 不代表通过完整科学流程的数值验收。
教程返回原坐标梯度作为检查输入，没有实现完整终点证书或 evidence 计算。

当前实验还使用终点实际模型的 Jacobian，检查局部 GN 驻点量
\(\tfrac12g^\mathsf TH_{\rm GN}^{-1}g\)；这是局部诊断，不是实际目标误差上界。
若还要评价 evidence，必须在当前终点按实际前向、当前强度和完整先验归一化计算，
不能用初始白化曲率或 L-BFGS 历史替代。

固定正则化强度下求 \(s,p\) 的 MAP，与外层选择正则化强度是两层任务。
不能直接把各个正则化强度也放进上面的裸 penalty 里一起最小化，代替完整 evidence 选优。

最后，源插值一阶导数的连续性会直接影响势修正的梯度。
白化和自动微分都不会自动消除插值折点；该问题的受控证据和边界处理见
[源插值与 L-BFGS 收敛经验](lbfgs_source_interpolation_lessons.md)。
