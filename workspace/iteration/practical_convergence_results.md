# J0946 双线性 source 插值的解稳定性与梯度停止条件

2026-11-06，`tinylens_gpu`，RTX 5080，Float64，JAX 0.7.1 / SciPy 1.17.1。

**旧双线性插值可以使联合 L-BFGS 得到大致稳定的恢复场。已有测试中的
“未收敛”经常是未通过光滑驻点检查，或在场已基本稳定后仍耗尽迭代预算，
不能直接解释成解值持续漂移或物理恢复失败。**

但它也有实际的精细求解限制：交叉初值能找到更低的同模型目标值，κ 整场仍有小幅
初值依赖。Hermite 与双线性结果的主要结构接近，整场并不数值相同；插值更换本身
改变了前向模型。下面分别衡量这些问题。

## 模型与复查范围

读取 [cubic_source_v1](cubic_source_v1/summary.json) 的全部 24 次固定权重结果，
并执行 6 次新拟合。新拟合固定最难的 `single_step_lens` 权重组：

- λs = 0.0003274148639653248，λ4 = 58605.388688316416，λ0 = 53622.67084530563。
- 2293 个数据像素，10000 个 source 系数，1134 个 potential 系数，802 个 κ core。
- 固定宏模型、source_scale、adaptive ρ、数据/noise/PSF/mask、nsub=4 和原 hard-M3
  白化参考；source 亮度系数与 potential 均更新。
- 原 L-BFGS-B 参数不变：maxiter=3000、maxfun=120000、maxcor=30、maxls=80、
  ftol=1e−15、gtol=1e−8。续跑重置 L-BFGS 历史，并保持原白化度量。
- 双线性采用固定零外环，Hermite 采用原 cubic 零延拓；两者外侧支撑分别为一格和
  两格。最早 bbox 外硬截零的版本另有亮度不连续，本次不将其与零外环版本混同。

原协议、源码和输入 hash 在新拟合前逐项核对；另核对的原实验输入与结果文件在
新拟合后均保持原 hash。输入和新实验设计见
[protocol.json](practical_convergence_v1/protocol.json)。

## 从旧终点继续优化时场几乎不再变化

这里 F 是 `0.5 × (χ² + source prior penalty + potential prior penalty)`。
表中变化均比较续跑后的终点和该次续跑的初点。

| 双线性续跑 | 新增步数 | F 的降低 | source 相对 L2 变化 | 全 core 最大绝对 Δκ | 图像差 RMS / noise | GN proxy 前 → 后 |
|---|---:|---:|---:|---:|---:|---:|
| 原 #18 终点 | 9 | 5.54e−9 | 6.16e−10 | 1.25e−9 | 1.26e−9 | .01348 → .02657 |
| 原 #22 终点 | 766 | 2.19e−6 | 2.64e−6 | 8.87e−6 | 7.15e−6 | .01953 → .01424 |

两次续跑都以 SciPy 的相对 F 变化条件正常停止，但仍未通过原 `proxy≤.001` 门槛。
尤其 #18 的场几乎不变，proxy 反而变大，说明该数值不是场恢复误差的直接量尺。

新进程从原 source-only 初值冷启动也得到相近解：2981 步后按相对 F 条件停止，
F=645.4157706195731，proxy=.03303。其相对原 #18 的 κ 整场 L2 差为 .214%，
source L2 差为 .0340%，图像差为 .000349 倍 noise。

这次冷启动保存的完整检查点进一步显示，场在原生退出前已经逐渐稳定。下表均相对
该次最终解；“F 差”只是相对这次终点的差，不是相对未知全局最优值的误差。

| 迭代步 | F 差 | κ 整场相对 L2 差 | 图像差 RMS / noise |
|---|---:|---:|---:|
| 100 | .008914 | 1.62% | .00273 |
| 500 | .000308 | .290% | .000348 |
| 1000 | .000135 | .137% | .000170 |
| 2000 | .0000106 | .0225% | .0000428 |
| 2900 | .000000329 | .000320% | .000000592 |

## 梯度检查为什么会与解稳定性脱离

只读复算原 `m3_source_fixed_v1` 的 #37/#38，两份 joint 数组最大差仅
1.54e−13，potential 最大差 2.12e−16，F 差 2.27e−13，κ 最大差 2.02e−14。
但双线性 physical gradient 的 L2 差为 **1059.58**，原 proxy 从 .00068466
变为 .0259284。在 cubic 前向下复算同一对数组，梯度 L2 差仅 1.24e−8。
详见 [branch_replay.json](practical_convergence_v1/branch_replay.json)。

格线探针也重新运行：两侧距离从 ±1e−5 格缩到 ±1e−9 格，双线性空间梯度差
始终约 107.2713；cubic 则从 .00551 缩到 5.51e−7。双线性的 C1 断言失败，
cubic 通过。

原因是 potential 改变取样位置 β：`∂image/∂p` 包含 source 的空间梯度。
双线性亮度连续，跨格线时空间斜率却可以跳变；因此同一个几乎不变的恢复场可以
对应不同的分支梯度和 GN 曲率。固定取样位置只优化 source 时没有这种格子切换。

一维 `F(t)=|t|` 就能说明判据的区别：t=0 是稳定极小值，左右导数却是 −1 和 +1。
这只是解释非光滑问题的例子，不是对本例多维终点最优性的证明。
`.5 gᵀ H_GN⁻¹ g` 是局部光滑二次模型的诊断，不是真实剩余 F 差的上界。

独立公开 kernel 与 AD 在 6 次新拟合终点的目标差不超过 2.28e−13，梯度相对差
不超过 9.92e−11；双线性终点的梯度相对差不超过 4.14e−13。没有发现分支内
导数实现错误。该跳变来自前向表示本身，不能只平滑返回梯度来消除。

## 稳定仍不等于唯一或已充分优化

把保存的 Hermite #19 解作为初值，**重新优化同一个双线性目标**，得到
F=645.3999123374757，比原 #18 的 645.4157188848478 再低 **.0158065**。
本次仍跑满 3000 步、proxy=.02060，未通过原验收。

相对原 #18，这个更低目标候选的 source L2 差约 .344%，κ 整场 L2 差约 **2.09%**，
最大 |Δκ|=.00848，图像差 .00260 倍 noise；κ 峰由 .21415624 变为 .21413251，
仅差约 **.0111%**。这证明原稳定终点还有一定初值/路径依赖，不能把所有优化限制
都归为梯度诊断失灵，也不能把原终点当作已证明的唯一 MAP。

反向将双线性 #18 作为 Hermite 初值，91 步回到 F=631.7358580281315，
proxy=6.71e−13；对原 Hermite #19，κ 最大差仅 4.39e−8、source 相对差 2.64e−8。
Hermite 原终点自身续跑 1 步，κ 最大变化 2.37e−9。新拟合的原生结果及全部检查点
分别保存在 [practical_convergence_v1](practical_convergence_v1/analysis.json)。

## 双线性与 Hermite 结果接近到什么程度

以下比较原受控实验的第一对同初值结果，两边均使用自身的前向渲染。
κ 相对 L2 范数以 Hermite 的 802 个 core 节点场为分母。

| 指标 | 双线性与 Hermite 的差异 |
|---|---:|
| κ 正峰位置 | 都是 (−.7, 1.0) arcsec |
| κ 正峰 | .21415624 对 .21186079，约差 1.08% |
| κ 整场相对 L2 差 | 9.65% |
| 全 core 最大绝对 Δκ | .02044 |
| source 节点相对 L2 差 | 2.96% |
| 预测图像差 RMS / noise | .02843 |

另外三组固定权重的 κ 整场相对差约 8.27%–10.74%，source 差约 2.15%–2.43%，
预测图像差约 .0316–.0325 倍 noise。主要形态接近，但峰值、整场与图像是不同的
一致性层次；每像素差低于 noise 也不等于整个数据集上的统计不可区分。

插值更换改变了目标函数，不能用 `F_bilinear≈645.42` 与 `F_cubic≈631.74` 的差
当作旧优化器剩余的优化误差。同一数组在两个目标下的值见
[cross_objectives.json](practical_convergence_v1/cross_objectives.json)。

场稳定也不能自动认证原 local-GN evidence：原 #37/#38 的场几乎相同，
diagnostic logE 仍差约 .01457，来自分支 Jacobian/曲率的变化。若后续按 evidence
选择超参数，还需单独检查这种评分敏感性。

## 对停止条件的含义和复现

对这批 J0946 结果，应分开记录“恢复场已稳定”“光滑驻点检查是否通过”和
“原生停止原因”。近似场恢复可以参考 F、source、κ、预测图像的尾段变化及
多初值差异；精细 MAP 和 evidence 则还需要更严格的一致性检查。C1 在本例显著
改善了后者和计算成本，但双线性未过原门槛不能等同于恢复场错误。

诊断入口为 [check_practical_convergence.py](check_practical_convergence.py)：

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python workspace/iteration/check_practical_convergence.py --require-stationary
# 预期失败：重放原验收条件，捕捉稳定场仍未过驻点门槛的现象。
python workspace/iteration/check_source_knots.py --method bilinear --require-c1
# 预期失败：旧空间梯度跨线不连续。
python workspace/iteration/check_source_knots.py --method cubic --require-c1
# 通过。
python -u workspace/iteration/check_practical_convergence.py --refine \
  --output workspace/iteration/practical_convergence_recheck
# 输出目录须为空；执行全部 6 次拟合，保留所有原生状态与检查点。
```

相关已有数学/接口回归另在独立 CPU 进程执行，8 项通过：

```bash
cd examples/tests
JAX_PLATFORMS=cpu pytest test_j0946_cubic_source.py test_j0946_iteration.py -q
```
