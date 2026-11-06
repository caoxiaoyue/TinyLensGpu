# J0946：C1 source 插值下重新最大化 lens 超参数 evidence

**结论：固定M3的source正则化后，C1模型经DIRECT＋两个起点的Powell精修，
保留的最佳合格候选δκ峰为0.200571，仍保持在.2左右。两次局部搜索均原生成功，
原候选＋5次额外冷启动均通过数值和重复性检查。** 该峰值相近不代表完整κ场相同。

## 保留候选与同模型基准

保留trial194的原始状态，没有用复跑中略高的分数替换。基准是在**同一C1迭代模型**
下，固定单步法的lens权重重新恢复的trial3；两者的source prior均固定M3值。

| 量 | 固定单步法lens权重 | 本次evidence保留候选 |
|---|---:|---:|
| λ4 | 58605.388688316416 | **437901.106694902** |
| λ0 | 53622.67084530563 | **8227.494422155389** |
| local-GN logE | 6298.167262409329 | **6304.584010896523** |
| δκ正峰 | .21186079004731495 | **.2005710308495934** |
| 峰位置，arcsec | (−.7,1.0) | **(−.7,1.1)** |
| χ² | 935.2272657354735 | 940.9833828689492 |
| 联合目标F | 631.7358580281315 | 628.7360763848817 |

相对基准logE增加 **6.41674849**，正峰降低 **5.33%**。峰位置移动一个.1 arcsec
potential网格间距。λ4变强、λ0变弱，改变了粗糙模态与较大尺度模态的相对惩罚。
Evidence同时权衡拟合、prior和曲率determinant，因此χ²略高而evidence更高并不矛盾。
不能用这个条件evidence差直接声称subhalo的Bayes factor或检测显著性。

最佳点距搜索边界至少2.15 dex，**没有撞超参数边界**。这是本次范围和预算内、
经数值认证的最高访问分数；有限预算DIRECT不提供全域最优证明。

## 外层与内层收敛

| 阶段 | native调用 | 新拟合 | 接纳 | native退出 |
|---|---:|---:|---:|---|
| 四个已知权重基准 | 4 | 4 | 4 | 各内层通过 |
| DIRECT | 129 | 129 | 119 | status1，maxfun软预算退出 |
| Powell：DIRECT后最佳起点 | 99 | 98 | 98 | **success，3轮** |
| Powell：单步lens权重起点 | 102 | 101 | 101 | **success，4轮** |
| 最佳点周围±.02 dex八点检查 | 8 | 8 | 8 | 全部分数较低 |
| 最佳点额外冷启动 | 5 | 5 | 5 | 全部内层通过 |

共 **345次实际恢复（340个去重权重点＋5次额外复跑）**，335次接纳、10次未通过。
两次局部搜索各有一次缓存调用，没有重复求解。DIRECT名义maxfun=120，实际129；
未触发160次显式硬上限，也未达到其len_tol停止条件，不能称DIRECT本身全局收敛。

两个Powell native最终返回值：

```
DIRECT后起点：log10(λ4,λ0) = (5.6413987139128, 3.9154764379265505)
              logE = 6304.58401011315
单步权重起点：log10(λ4,λ0) = (5.64152393568727, 3.915153087396503)
              logE = 6304.584010137809
```

两个返回分数相差 **2.47e−8**，权重差不超过约.075%。保留它们访问过的trial194，
分数仅比native返回点高约8e−7；未把访问最优与native最终端点混为一个状态。
八个邻近点的logE均低 **.00551–.01564**，没有触发额外精修。

10个未接纳点全部来自DIRECT探索，均达到3000步上限，无factor异常；其中部分
额外GN检查已较小，但native未成功，仍按预先规则保留为不合格。失败不等于物理
恢复必定错误，也不将其diagnostic evidence正式评分。C1在本次高分区间表现稳定，
不代表任意极弱正则化下都能在固定预算内通过。

## 最佳点的冷启动重复性

原候选和5次额外复跑，共 **6/6通过**。额外复跑均恢复同一组超参数，未热启动，
source亮度像素仍更新。

- L-BFGS：**134–136步**；GN诊断 **1.45e−12–3.42e−12**，门槛仍为.001。
- δκ峰范围：**.20057103034248724–.20057103195677825**。
- 全部6个场的全core最大两两绝对差：**1.72076e−8**，门槛1e−4。
- local-GN logE范围宽度：**2.96030e−7**，门槛.01。
- 因此原重复协议通过，sidecar为`maximum_evidence/validation.json: passed`。

最佳原始点内层约 **1.110秒**；当次白化.149秒、终点GN/evidence核验.647秒，
常规单次合计约 **1.906秒**。额外复跑内层中位数 **1.085秒**，加额外QR审计后
fit总计中位数2.181秒；外部fresh public梯度核对与导出IO不属于内层计时。
整个345次实验含准备、优化、审计和导出约 **1105.46秒（18.42分钟）**，
不含后续只读分析和出图。DIRECT约585.55秒，两次Powell约228.84、238.77秒。

## 峰保持，但整个κ场有明显重分布

本次最佳候选的δκ最小值 **−.163864**，core RMS **.0490294**；基准最小值
−.154579、core RMS .0465242。二者完整802个core节点之差：

```
RMS(δκ_selected − δκ_baseline) = .04542054
max |δκ_selected − δκ_baseline| = .16967583
```

因此不能仅凭峰都在.2附近就称整场一样。图中最优权重的正负修正更偏向较大尺度，
主要正峰位置近似保持，宽度和其他位置的修正明显改变。这是**更换超参数后的场差**，
不是同一超参数冷启动的不稳定；后者最大差仅1.72e−8。

最优拟合归一化残差RMS约 **.64060**，最大绝对值约 **2.21964**。它们是含固定
prior的恢复诊断，χ²/2293≈.41037不等于用有效自由度计算的reduced χ²。J0946没有
已知injected κ真值；当前结果确认的是条件优化的数值稳定，尚未确认真实subhalo
质量、密度剖面、唯一性或检测显著性。

## 验证记录

- 独立Hermite参考、节点语义、C1连续、解析/AD梯度、PSF/子像素转置、dense完整
  耦合evidence/白化、已有boundary和handoff检查：**13 passed（15.29秒）**。
- 9次真实数据fresh public梯度/目标核对：最大目标差2.28e−13、梯度相对差1.75e−10。
  白化二次型误差≤2.89e−15，QR/Cholesky logdet差≤3.19e−12。
- **345个导出全部核对**相同source λ、scale、bbox、data/noise/mask、节点拓扑和初值。
  所有被测代码、M3输入和旧结果hash未改；source像素未冻结。正式evidence仅在接纳时
  写入，原始选中状态不被复跑覆盖。库和旧实验入口未改，只新增workspace实验脚本。
- 图为trial194原始状态及trial3同模型基准；全部345条记录保留在source CSV，
  数值原图阵列另存NPZ，未平滑κ场或隐藏负值。最终图文/对齐/碰撞检查见figure QA。

## 实验定义

2026-11-04，`tinylens_gpu`、RTX 5080、Float64。这里研究固定宏模型、lens light、
source 网格及自适应 prior 条件下的 lens 超参数；source 亮度像素仍自由更新。

- λs 固定 M3 使用值 **0.00032741486396532473**，实际 log10 往返为
  **0.0003274148639653248**；ρ固定 **5.530494995245579**。
  M1 模板、source_scale、first-order prior 及其归一化保持不变。
- Source 为原100×100个亮度节点，C1 cardinal bicubic 插值，外侧节点固定零；
  potential 仍1134个系数、802个core，间隔.1 arcsec；内层联合优化11134个系数。
- 数据2293个像素，PSF、nsub=4、noise、mask、source bbox、宏模型和减光输入均
  来自同一个M3 handoff。不重跑smooth，不新增subhalo参数，不做source rescaling。
- λ4 D4ᵀD4＋λ0 I：D4沿用完整pixel-unit四阶差分；λ0作用于全部potential节点。
  没有改正则化定义、新增jitter或只惩罚某个gauge子空间。
- 所有候选从上一轮C1对照使用的**同一个冻结source初值＋零potential**冷启动。
  不从相邻权重的结果热启动；白化参考仍为原hard-bilinear M3状态。
- L-BFGS-B维持maxiter=3000、maxfun=120000、maxcor=30、maxls=80、ftol=1e−15、
  gtol=1e−8。使用JAX自动微分；数值验收为native success、有限/SPD、GN诊断≤.001。
- Evidence采用终点**完整source–potential耦合**的local-GN Gaussian approximation，
  保留source和lens prior归一化；不是exact evidence或完整Hessian Laplace。

## 搜索和验证协议

只优化log10 λ4与log10 λ0。沿用原宽边界：

```
log10 λ4 ∈ [−1.2051919733248866, 7.794808026675113]
log10 λ0 ∈ [ 0.9378228269292088, 6.937822826929208]
```

先在同一C1模型下重新测量四个已有权重检查点，包含单步法lens权重的对照。
随后原版DIRECT（locally_biased=False）名义maxfun=120、maxiter=50、len_tol=.002，
另设160个新拟合的硬上限。名义maxfun是软预算，实际次数与原生退出状态单独保存。

之后进行两个独立起点的bounded Powell：DIRECT后最高合格分数点、单步法lens权重。
每次maxfev=160、maxiter=12、xtol=.002、ftol=1e−8。访问过的最高合格分数和native
最终返回值分别保存；拒绝点保留diagnostic evidence，但正式分数为空，不参与最大化。

最佳点周围±.02 dex做8点二维邻域检查；若有超过.001的evidence改善，再精修。
保留的原始候选再做**5次**相同条件冷启动，不以其中更高的分数覆盖原始候选。
原候选＋5次复跑共6次全部通过，并满足logE范围≤.01和全core两两最大κ差≤1e−4，
才记为通过重复性验证。这是同一数据的技术重复，不是独立噪声模拟或后验误差。

每个候选保留原生优化状态、目标函数历史、joint状态和完整恢复图。强prior初次及
最终复跑另外审计白化二次型、独立QR determinant；基准和最终复跑使用实际λ0的
fresh public problem核对解析梯度与AD。所有source、输入及旧结果hash均检查。

本轮与以前双线性探索相比，同时改变了插值与搜索预算，不能把全部差异归因于
单一改动。所有候选按合格evidence选择，κ峰是否接近.2不参与选择。

## 复现入口

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python workspace/iteration/run_cubic_evidence.py --output workspace/iteration/cubic_evidence_new
python workspace/iteration/analyze_cubic_evidence.py workspace/iteration/cubic_evidence_new
python workspace/iteration/plot_cubic_evidence.py workspace/iteration/cubic_evidence_new
```

脚本：[运行](run_cubic_evidence.py)、[数值核对](analyze_cubic_evidence.py)、
[出图](plot_cubic_evidence.py)。正式运行目录：`cubic_evidence_v1/`。

结果：[完整数值汇总](cubic_evidence_v1/analysis.json)、
[选中点重复认证](cubic_evidence_v1/maximum_evidence/validation.json)、
[完整场差核对](cubic_evidence_v1/postfit_field_comparison.json)、
[拟合和κ图](cubic_evidence_v1/figure/diagnostics.png)。
