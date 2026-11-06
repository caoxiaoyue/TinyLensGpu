# J0946：固定 M3 source prior 的 C1 插值对照

2026-11-04；`tinylens_gpu`；RTX 5080；Float64。

**本次固定权重实验支持：把 source 双线性插值换成一阶导数连续的插值，足以消除
之前复现的内部格线梯度跳变，并显著改善这四组权重下 L-BFGS 的收敛和重复性。**
新插值 12/12 次通过原验收，双线性对照 7/12；四组新插值均通过重复门槛。
这不是全域超参数搜索结果，也不证明真实 subhalo 的幅度已经确定。

## 严格控制的条件

- 固定 λs = **0.00032741486396532473**，实际 log10 往返为
  **0.0003274148639653248**；adaptive ρ = **5.530494995245579**。
  它们是 M2 后验中位数、随后固定用于 M3 的值，未重新估计。
- M1 模板、自适应 `source_scale`、first-order source prior 和归一化不变。
  Source 为原来的 100×100 亮度节点；**10000 个亮度系数仍自由更新**。
- Macro、lens light、减光后的输入、2293 个数据像素、PSF、noise、mask、nsub=4、
  source bbox、1134 个 potential 节点/802 个 core、0.1 arcsec potential 间隔不变。
  沿用原 λ4 D4ᵀD4＋λ0 I；没有新增 gauge、jitter 或 source rescaling。
- 两种插值使用**完全相同的起点数组**：上一轮冻结的 zero-ring source-only MAP，
  potential=0。该初值在旧双线性模型下求得；本次不为 cubic 重新求 source-only MAP。
  每次原样冷启动，不从上次恢复热启动，不筛选重复结果。
- 两种插值共享原 hard-bilinear M3 source/zero-potential 的白化参考；每次重新构造
  相应 lens λ 的数值白化因子，没有同步改为实际初值白化。
- L-BFGS-B：maxiter=3000、maxfun=120000、maxcor=30、maxls=80、ftol=1e−15、
  gtol=1e−8，SciPy 1.17.1 / JAX 0.7.1。前向和梯度均用 JAX Float64。
- 原验收：native success、有限/SPD、`.5 gᵀ H_GN^-1 g ≤ .001`。
  原重复门槛：全部通过、diagnostic logE 范围≤.01、全 core 最大两两 |Δκ|≤1e−4。
  这里的 GN 量是驻点诊断，不是真实 objective gap 的上界。

四组 lens 权重均在观察新 cubic 结果前按已有结果固定。每组每方法 3 次同初值重复，
合计 4×2×3=24 次真实拟合。**没有 DIRECT/Powell 外层搜索，也没有重跑 M3。**

| 固定 lens 权重来源 | λ4 | λ0 |
|---|---:|---:|
| Strong anchor | 31622776.60168379 | 3162277.660168379 |
| 旧迭代 lens 权重 | 737829.5447469671 | 1570503.482633853 |
| 上轮固定 M3 source 后保留的 lens 权重 | 553824.033132111 | 3162277.660168379 |
| 单步法 lens 权重 | 58605.388688316416 | 53622.67084530563 |

“单步法 lens 权重”只复用 λ4/λ0；λs 仍为本次固定的 M3 值，并非单步法所选 λs。

## 新 source 表示

采用张量积三次 Hermite 插值，节点导数为中心差分，即 Catmull–Rom / Keys a=−.5。
每条光线最多引用 4×4 个节点，source 系数仍是原节点亮度，**不是另换含义的
B-spline 控制系数**。整数节点上精确插值，不新增斜率或其他未知数。

以单位网格内 t∈[0,1] 为例，四个节点权重为：

```
w[-1] = −t/2 + t² − t³/2
w[ 0] = 1 − 5t²/2 + 3t³/2
w[ 1] = t/2 + 2t² − 3t³/2
w[ 2] = −t²/2 + t³/2
```

相邻格子在端点的亮度和一阶导数匹配。全部外侧节点固定为零，有限支撑在原
source 边界外两个网格间距处连续归零且斜率为零；双线性 zero-ring 的支撑为一个
间距。这个支撑宽度变化是所选 cubic 零延拓定义的一部分，未增加拟合节点或参数。
三次权重可为负并产生局部过冲；沿用原 signed source，不另加正值裁剪。

前向、source 转置、potential Jacobian、独立解析梯度、终点 evidence 全部采用
同一新插值。Evidence 仍包含完整 source–potential 耦合和相同节点 prior normalization，
使用 endpoint local-GN Gaussian approximation，不是 exact evidence 或完整 Hessian Laplace。
Hermite 的 C1 定义可参见 [SciPy 官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.html)；
实际 GPU 实现是独立 JAX 稀疏算子，不在优化过程中调用 SciPy 插值器。

## 内部格线机制：先复现，再检验

在上一轮失败 repeat38 的保存状态中选取距内部格线最近的光线 #4955，x 方向
跨线；不重新优化。将跨线偏移从 ±1e−5 格缩到 ±1e−9 格：

| 插值 | 1e−5 格的梯度差范数 | 1e−7 格 | 1e−9 格 |
|---|---:|---:|---:|
| 原双线性 zero-ring | 107.271289 | 107.271289 | 107.271289 |
| C1 cubic | .005511912 | .0000551196 | .000000551194 |

旧方法的 C1 断言明确失败，新方法通过；亮度在两者中都连续。进一步只读复算旧
repeat37/38 的两个 joint 数组：max|Δq|=1.54e−13，max|Δpotential|=2.12e−16。
在相同实际 λ 下，双线性 physical gradient 的 L2 差约 **1059.58**；cubic 仅
**1.25e−8**，相对差约 **6.96e−13**。旧 witness 并非 cubic 的最优点，该复算只检测
映射分支敏感性，不将它当作 cubic 拟合或评分结果。

## 24 次端到端恢复的结果

“通过”同时要求 native 和独立 GN 门槛；κ 为 potential correction 的 δκ，非总 κ。

| Lens 权重组 | 双线性通过 | Cubic 通过 | 双线性步数范围 | Cubic 步数范围 | 双线性 κ峰 | Cubic κ峰 |
|---|---:|---:|---:|---:|---:|---:|
| Strong anchor | 3/3 | **3/3** | 13–20 | 17 | .0073954 | .0072766 |
| 旧迭代 | 1/3 | **3/3** | 64–159 | 26 | .045256–.045259 | .0449881 |
| 上轮保留 lens 权重 | 3/3 | **3/3** | 20 | 22 | .0376663 | .0369856 |
| 单步法 lens 权重 | 0/3 | **3/3** | 3000 | 95–96 | .214141–.214164 | .2118607893–.2118607901 |

新插值四组 GN 诊断范围约 **1.68e−14–9.28e−13**，远低于不变的 .001 门槛。
双线性旧迭代组 `.000374–.005484`，单步 lens 组 `.013389–.019526`。
本次双线性上轮保留点的三次碰巧全部通过，不抹去上一轮有重复失败的记录；旧
37/38 原数组仍按原结果保留，格线复算也使用这些原数组。

### 时间：计时范围必须一致

均为该组 3 次中位数；正式计时前显式 warm-up 前向/梯度和坐标解码。

| Lens 权重组 | 双线性内层 | Cubic 内层 | 双线性单次总计 | Cubic 单次总计 |
|---|---:|---:|---:|---:|
| Strong anchor | .302 s | **.136 s** | 1.044 s | .928 s |
| 旧迭代 | 1.196 s | **.201 s** | 1.936 s | .992 s |
| 上轮保留 | .707 s | **.172 s** | 1.447 s | .965 s |
| 单步法 lens 权重 | 31.152 s，未通过 | **.766 s，通过** | 31.895 s | **1.561 s** |

单次总计包括当次白化因子、L-BFGS、终点完整耦合 GN/evidence 核验，**不含一次性
加载、engine 准备/JIT、独立审计和导出 IO**，所以不能将它叫作新进程启动的总耗时。
单步 lens 权重的 cubic 分解约为：白化 **.150 s**、内层 **.766 s**、终点核验 **.645 s**。
三个 engine 一次性准备约 **5.30 s**，另有两模型白化梯度 warm-up 共约1.71 s；这些
不是单次求解开销。整个 24 次实验连同准备、独立核对和导出约 **146.00 s**。

最困难组内层约快 **40.7 倍**，这是“原方法跑满预算未过门，新方法通过”的对照，
不是两个方法收敛到同一目标最优解的速度比。新插值改变了前向表示，F 也相应变化。
上轮保留点 cubic 虽22步略多于双线性20步，但函数/梯度调用24次对106–107次；
单步 lens 组则102–103次对4192–4325次，显著减少了格线附近的线搜索消耗。

### 重复稳定性

| Cubic 权重组 | GN 门全部通过 | 全 core 最大两两 κ 绝对差 | logE 范围宽度 | 原重复门槛 |
|---|---|---:|---:|---|
| Strong anchor | 是 | 3.30e−17 | 0 | 通过 |
| 旧迭代 | 是 | 4.79e−15 | 9.09e−13 | 通过 |
| 上轮保留 | 是 | 8.32e−13 | 5.46e−12 | 通过 |
| 单步 lens | 是 | **5.71e−9** | **2.13e−7** | 通过 |

这些是同一数据、同一初值、同一进程内重建因子的技术重复；不是独立噪声数据，
不是不同初值的吸引域研究，也不是后验不确定度。现有结果不能保证所有 λ 或宏模型
条件下都收敛。

## 对 κ 和物理恢复的解释

单步 lens 权重下，两种表示的主要峰均在 **(−.7, 1.0) arcsec**，cubic 峰约 .211861，
相对本次 bilinear 峰小约1.1%；其三次整场差远小于原重复门槛。Cubic χ²≈935.2273，
双线性未过门终点约962.26–962.29；前向表示不同，不能把这直接解释成同一目标的
优化差距。以各组三次平均场比较，两方法单步 lens 组全 core 差异 RMS≈.004494、
最大绝对差≈.02038；不能只看峰很接近就称整个场完全一样。

原较强的上轮保留 lens 权重下，cubic 峰仍只有 **.036986**，没有自动变成 .21；
峰位置从双线性 (−.7,1.1) 变为 cubic (−.7,1.0)。旧迭代权重 cubic 峰约 .044988。
这说明**格线引发的内层收敛困难得到显著改善，正则化强度对恢复幅度的影响仍在**。
本次未重新最大化 evidence，旧表示下搜索到的 lens λ 不能冒充新表示下最优值。
J0946 没有已知 injected κ 真值；这里尚未证明 subhalo 质量、检测显著性或唯一物理解。

## 验证与产物

- `examples/tests/test_j0946_cubic_source.py`：独立 SciPy Hermite 张量积参考，原节点亮度，
  内/外格线 C1 与 AD，含 PSF+nsub 的完整梯度/转置恒等式，相同 prior，白化及 dense
  full-coupled evidence 和 GN 诊断。加既有边界/handoff 检查共 **13 passed**。
- 真实数据每组首个重复额外使用实际 λ0 的全新 public problem 做解析核对；最大
  objective 差 **2.28e−13**、梯度相对差 **1.07e−10**。强 prior 首次还做 QR/determinant
  和白化二次型核验。没有把 saved-M3 的双线性 Jacobian 用于 cubic 的终点 evidence。
- 24 次导出逐项核对相同 source 初值、source_scale、bbox、mask、noise、数据和 λ；
  source 亮度确实自由变化。全部代码/输入 hash 未改，旧结果未覆写。
- 新实现只在 `workspace/iteration/`；库默认插值、正式 `pt.py`、`lbfgs.py` 未改。
- κ图为两组预声明案例的 repeat0，包含未通过的双线性终点，未按通过/峰值挑图。
  所有24次记录另存CSV。PDF图元/字体、最终面板对齐和碰撞均核验，见 figure QA。

运行与复核：

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python workspace/iteration/check_source_knots.py --method bilinear --require-c1  # 预期失败：复现原折点
python workspace/iteration/check_source_knots.py --method cubic --require-c1
python workspace/iteration/run_cubic_source.py --output workspace/iteration/cubic_source_new
python workspace/iteration/analyze_cubic_source.py workspace/iteration/cubic_source_new
python workspace/iteration/plot_cubic_source.py workspace/iteration/cubic_source_new
```

正式运行目录：[cubic_source_v1](cubic_source_v1/)。
[运行脚本](run_cubic_source.py) · [独立插值算子](source_cubic.py) ·
[核对脚本](analyze_cubic_source.py) · [完整数值汇总](cubic_source_v1/analysis.json) ·
[原状态梯度复算](cubic_source_v1/saved_endpoint_gradient_replay.json) ·
[κ 对比图](cubic_source_v1/figure/kappa.png)。
