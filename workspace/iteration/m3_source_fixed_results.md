# J0946：固定M3的source正则化，只搜索lens超参数

2026-11-04，`tinylens_gpu`，RTX 5080，Float64。

**结论：固定M3的source超参数后，程序能完成联合迭代和二维evidence探索，但没有
解决内层/候选重复认证的不稳定。** 单步lens权重下κ峰约.214，仍未过原门槛；
搜索保留的合格候选κ峰约.0376，三次额外复跑仅2次通过，整体`repeat_gate_failed`。
这不是“所有解都无法接近目标极小值”的证明，也不是可靠subhalo检测结论。

## 按用户确认固定什么

固定source正则化参数，**source亮度像素仍更新**。沿用M3实际使用的值：

- λs = **0.00032741486396532473**；保存的自然对数为−8.024282494041525。
- adaptive ρ = **5.530494995245579**，M1模板及完整自适应source_scale图固定。
- first-order prior、100×100 source、source bbox、mask/noise/PSF、nsub=4不变。

严格说这组超参数由M2后验中位数确定，然后固定用于M3；不是在本次重新估计。
宏模型仍为M3保留样本#21342的最大likelihood代表点，不重跑smooth或减光。
保留2293个数据像素（包括原8个noise=1e8像素）、1134个potential系数和802个core。
内层仍是10000 source＋1134 potential的联合MAP；减少的是**外层超参数维数**，
从λs/λ4/λ0三维变为仅log10 λ4、log10 λ0二维。source没有从参数或evidence积分中删去。

主模型使用已验证的固定零环source边界；另在单步lens权重下重跑原硬边界作对照。
零环不增加自由参数，prior仍在原节点上，未重新调整source尺度或normalization。
白化仍以保存的硬边界M3状态为参考，主模型共享同一参考metric，未同时改成实际
起点白化或引入新的插值/优化器。其余物理条件保持与前一轮边界实验一致。

λs在实际Float64转换后为.0003274148639653248，差异只是log10往返舍入；所有39次
恢复均使用同一个值。源自适应权重、source bbox、数据和noise也逐文件核对相同。

## 初始化、验收与预算

每种边界只在固定M3 λs、p=0下计算一次signed source-only MAP，保存该完整数组；
之后所有候选和重复从同一数组、零potential重新开始。source初值不依赖lens λ，
不从邻近候选热启动。白化数值因子仍每候选重新构造。

L-BFGS-B保持maxiter=3000、maxfun=120000、maxcor=30、maxls=80、ftol=1e−15、
gtol=1e−8。统一验收仍是native success、有限/SPD和`.5 gᵀH_GN^-1g≤.001`。
evidence保留完整source–potential耦合及归一化，采用终点local-GN高斯近似。
该梯度诊断不是真实objective gap或evidence误差上界，尤其不构成折点处的完整
非光滑最优性判定。失败点保存raw状态及diagnostic evidence，不正式评分。

预先声明的选中点重复目标：所有尝试通过内层门槛，diagnostic logE范围宽度≤.01，
相对原选中场全core max|Δκ|≤1e−4；原选中点另独立冷启动复跑3次，不挑最好的一次。

先做三组lens权重各3次主模型恢复，再做原硬边界的单步lens权重3次，共12次探测。
外层保持原完整lens bounds：log10 λ4约[−1.20519,7.79481]，log10 λ0约[.93782,6.93782]。
DIRECT最多3轮、名义及显式新拟合预算16；有界Powell maxfev=16、maxiter=3，另设
新拟合预算16。若显式预算截断native调用，会单独标记，不冒充原生收敛。

## 固定lens权重的重复结果

source超参数均为上述同一M3值：

| 主模型的lens权重来源 | λ4 | λ0 | 通过 | κ峰范围 | GN诊断范围 |
|---|---:|---:|---:|---:|---:|
| 已声明强prior检查点 | 3.1623e7 | 3.1623e6 | 3/3 | .00739541–.00739547 | 3.18e−8–9.01e−8 |
| 既有单步lens强度 | 58605.3887 | 53622.6708 | **0/3** | **.214153–.214171** | **.01954–.02614** |
| 旧迭代搜索保留点的lens强度 | 737829.5447 | 1570503.4826 | **0/3** | **.04525851** | **.00126258** |

单步lens权重下3次都达到3000步，内层中位数约30.72秒。旧迭代lens权重下，3次
native都在约64步按relative-F停止、约1.23秒，但GN诊断略高于原门槛，仍不接纳。
强prior检查点约13步、.34秒通过，但把局部修正压得很小。

原硬边界单步lens权重对照也0/3通过，κ峰.214129–.214161，GN诊断.01266–.02270，
一次native rel-F、两次达到3000步。故此次失败不能仅归为新零环模型的影响。
两种边界下这组目标F约645.415；不同lens λ的F含不同penalty，不能直接用其大小
判断哪个超参数更优，应按有效且可重复的evidence比较。

## 二维DIRECT＋Powell探索

- DIRECT实际9个新点、3轮，9点都未通过内层门。原生status=2、maxiter退出，
  `nfev=9`，耗时约200.28秒；不是外层收敛。返回cost=1e30是无可靠评分的惩罚。
- Powell实际16次调用，1次缓存、15个新点，其中10点接纳，耗时约34.88秒。
  原生status=1、maxfev退出、nit=0。预算主要耗在λ4方向，λ0没有完成局部精修。
  native最终返回原起点，但外部日志保留其访问过的最高合格分数。
- 两阶段都没有触发额外的显式新拟合cap。native退出原因与访问过的最佳候选
  分别保存，不能将后者称为全域最大evidence或二维精修已完成。

搜索保留trial#31：

```
λs = 0.0003274148639653248（固定）
λ4 = 553824.033132111
λ0 = 3162277.660168379
logE_GN = 6230.464380093842
GN诊断 = 0.0005628567660307251
δκ峰 = 0.03764255239264042，位置(−.7″,1.1″)
```

它单次通过，在固定source超参数下找到可运行区域；不是充分复现的最优结果。
原宏模型nested logZ包含位置惩罚且采用不同determinant，不能据此算Bayes factor。

## 选中候选的重复认证未通过

| 尝试 | native success | 步数 | GN诊断 | diagnostic logE | 接纳 |
|---|---|---:|---:|---:|---|
| 搜索内trial31 | 是 | 200 | .00056286 | 6230.464380 | 是 |
| 额外复跑36 | 是 | 20 | .00068466 | 6230.466312 | 是 |
| 额外复跑37 | 是 | 20 | .00068466 | 6230.466312 | 是 |
| 额外复跑38 | 是 | 20 | **.02592838** | **6230.451744** | **否** |

三次额外复跑κ峰均约.03766627。包括原选中点的4个场中，diagnostic logE范围
宽度.0145681，max|Δκ|=.000133820；均略超过预先声明的.01/1e−4重复性目标，
而最后一次内层GN门也失败。因此sidecar明确为`repeat_gate_failed`。
trial38的正式log_evidence仍为null，没有用其诊断分数取代失败状态。

共39次真实拟合：12次探测、9次DIRECT新点、15次Powell新点、3次选中点复跑；
15次接纳、24次拒绝，无factor异常。整个实验约447.97秒（7.47分钟），不含后续
只读核对。39次的source亮度都允许变化；保留候选source相对M3 L2差约3.86%，
单步lens权重组约14.04%。全探索范围约1.96%–67.24%，后者包括不合格强扰动点，
不能当作物理可信source变化范围。

## 只读分支核对：同一候选为何证书跳变

原始拟合冻结后，对36/37/38的保存数组重新核对，没有运行优化或替换终点：

- 37→38的最大joint系数差1.54e−13、最大potential系数差仅**2.12e−16**。
- 只改变了**1条光线**所选的内部双线性格子；最近光线距离格线仅约1.42e−14格。
- 复算F在约2e−13以内一致，但保存的GN诊断从.00068466变为.02592838。
- 每个状态的AD梯度与独立 `Jwᵀ rw + Qq` 核对误差约5.7e−13–3.7e−12，
  没有发现分支内的梯度公式错误。

这把该候选的证书变化定位到了内部插值折点附近：固定source超参数没有改变
双线性插值斜率会跳变的性质。它支持“非光滑目标/局部GN近似仍影响数值认证”，
不证明这39个终点都没有接近实际极小值，也不能用不稳定的局部evidence来确认
物理subhalo幅度。更连续的插值或适合该目标的认证方式需作为后续独立实验。

## 核验与复现

复用的9项针对性数学/边界/handoff回归通过。全部39个实际导出都核对λs、ρ、
source_scale、bbox、mask/noise/data与M3输入保持一致，source初值与冻结数组一致；
审计候选最大解析/AD目标误差2.28e−13、梯度相对误差6.10e−11。原source、单步、
旧迭代及smooth文件hash未改。二维预算异常的native传递也用独立CPU探针核对。

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python workspace/iteration/run_m3_source_fixed.py --output workspace/iteration/m3_source_fixed_new
python workspace/iteration/analyze_m3_source_fixed.py workspace/iteration/m3_source_fixed_new
python workspace/iteration/audit_m3_source_branches.py workspace/iteration/m3_source_fixed_new
```

正式结果目录为`m3_source_fixed_v1/`。被测程序及依赖快照、协议、初值、每个候选
原生状态/历史/恢复场都保留；没有为得到通过结果改门槛或重新挑终点。

- [汇总](m3_source_fixed_v1/analysis.json)、[协议](m3_source_fixed_v1/protocol.json)
- [选中候选验证状态](m3_source_fixed_v1/maximum_evidence/validation.json)、[分支核对](m3_source_fixed_v1/postfit_branch_audit.json)
- [运行脚本](run_m3_source_fixed.py)、[核对脚本](analyze_m3_source_fixed.py)、[分支核对入口](audit_m3_source_branches.py)
