# J0946：C1 source 下联合优化 λs、λ4、λ0

**结论：释放λs后，主实验保留解的δκ峰为0.17433，比固定M3 λs时的0.20057降低
13.1%；峰位置仍为(−.7,1.1)arcsec。数值恢复和同权重重复性稳定。** 该结果仅针对固定macro和自适应
prior形状的本次近似evidence优化，不是subhalo真实质量或全域最优的证明。

## 三维保留候选与二维对照

两者均在本次C1模型、相同source初值和相同数据下重新测量；对照为上轮固定λs
的最优访问点，不是另换单步前向模型。主实验trial497的原始状态保留不动。

| 量 | 固定M3 λs、优化lens | 释放λs、联合优化 |
|---|---:|---:|
| λs | .0003274148639653248 | **.0002532641285041916** |
| λ4 | 437901.106694902 | **538677.9228218549** |
| λ0 | 8227.494422155389 | **10572.677018080045** |
| local-GN logE | 6304.584010902424 | **6307.427179447846** |
| δκ正峰 | .20057103084552513 | **.17433177392076282** |
| 峰位置，arcsec | (−.7,1.1) | **(−.7,1.1)** |
| χ² | 940.9833828522424 | 920.2926875641397 |
| 联合目标F | 628.7360763848819 | 605.0380047097701 |

logE增加 **2.84316855**。λs减小 **22.65%**，λ4和λ0分别增大 **23.01%**、**28.50%**：
source惩罚减弱而potential惩罚增强，较灵活的source与较小的potential修正取得新的
权衡。F对应不同权重，不能将其差直接解释为同一目标的优化误差；χ²和evidence
则分别报告各自的拟合与复杂度权衡。没有通过κ幅度筛选候选或更改验收门槛。

最佳点距各边界均至少1.89dex，未撞界。Source λs实际访问范围约1.52e−5至.00705，
区别于预先声明的更宽边界；有限DIRECT不保证遍历全部区域。

## 收敛、预算与补充确认

| 阶段 | callback次数 | 新拟合 | 通过 | native退出 |
|---|---:|---:|---:|---|
| 四组基准 | 4 | 4 | 4 | 各内层通过 |
| DIRECT | 179 | 179 | 171 | status1，maxfun软预算 |
| Powell：DIRECT后最佳点起步 | 103 | 102 | 102 | **success，3轮** |
| Powell：单步三项权重起步 | 240 | 239 | 239 | **status1，maxfev，5轮** |
| 三维±.02dex邻域 | 26 | 26 | 26 | 所有分数更低 |
| 额外相同权重冷启动 | 5 | 5 | 5 | 全部通过 |

主实验共 **555次恢复，547次接纳，8次未通过**；550个去重权重点，加5次明确复跑。
所有未接纳点来自DIRECT，均耗尽3000步；无factor异常。两次Powell各复用一次缓存。
DIRECT名义maxfun160，实际179，未触发220次显式硬上限。它未达到len_tol停止条件，
不能称其全局收敛。

第一起点native结束logE=6307.427163458743，κ峰.1744786。第二起点native返回
logE=6307.4271793918，访问过的trial497略高约5.6e−8；它虽到达几乎同一解附近，
仍因maxfev退出。两起点到达区域的分数差约1.60e−5，不能把第二次预算退出改写为成功。

为进一步核对，主实验结束后，**另建独立确认运行**，从trial497权重重新开始bounded
Powell，保持所有物理条件和相同冷启动，maxfev120、xtol.002、ftol1e−8；不改写主实验。
该确认native **success，status0、1轮、27次调用，其中26次新恢复全部通过**。
其native最终logE=6307.427177023412，比保留点低约2.42e−6；所有新访问点均未超过
主实验保留点。确认耗时 **71.09秒**，`confirmed=true`。主实验原始候选、重复认证
和“第二起点预算退出”记录完整保留，没有用确认结果掩盖原生退出。

主实验26个邻近点的logE均低 **.00423–.13750**，未触发额外改善精修。
保留的是主实验最高合格访问分数；所有后续确认也未发现更高分，不声称严格全域最优。

## 最终权重的重复稳定性

Trial497＋5次相同条件冷启动，**6次全部通过**：

- L-BFGS步数131–135，GN诊断1.05e−12至3.51e−12，门槛仍为.001。
- δκ峰范围 **.1743317571422823–.17433177392076282**。
- 全802个core节点的最大两两κ差 **3.47343e−8**，门槛1e−4。
- local-GN logE范围宽度 **6.15030e−7**，门槛.01。
- 因此sidecar为`maximum_evidence/validation.json: passed`。

这些是相同观测、相同初值、重新构造白化因子的技术重复，不是独立观测、不同
source初值的吸引域测试或后验置信区间。补充确认改变的是外层权重，和这些同权重
复跑分开记录。

原保留点：白化.151秒、内层 **1.036秒**、终点GN/evidence .648秒，常规fit总计
**1.834秒**，不含一次性准备、额外fresh public审计和导出IO。5次复跑内层中位数
**1.062秒**；含额外QR审计的fit总计中位数2.157秒。

主实验约 **1576.70秒（26.28分钟）**；补充确认71.09秒。两者合计约27.46分钟，
不含后处理和绘图。三维释放λs没有增加内层11134个未知数，增加的是外层搜索维度。

## 完整κ场与拟合

主峰位置保持；完整场也存在变化，不能仅凭峰值判断恢复完全相同。对照与保留场：

```
RMS(δκ_joint − δκ_fixed_source) = .00912556
max |δκ_joint − δκ_fixed_source| = .02771319
fixed-source core RMS/min = .0490294 / −.163864
joint-three  core RMS/min = .0419503 / −.137024
```

图中主要修正形态大致保持，正负修正幅度整体减弱；右下差值图单独标注色标，
没有隐藏负值、平滑或按峰选择复跑。与同权重重复差3.47e−8相比，这些更大差异来自
超参数重选，而非同一个解数值不稳定。模型的真实subhalo幅度仍需别的物理验证。

## 完整性与交付

- 所有 **555个导出**均核对相同data/noise/mask、source_scale、source_bbox、节点拓扑
  和source初值。λs/λ4/λ0逐候选核对实际值；ρ和prior形状固定。原始选中状态的
  NPZ/JSON与trial497字节一致，未由更高分复跑替换。
- 新增source λs变化下的小矩阵测试，加旧C1/边界/handoff测试，**16 passed**。
  9次真实候选的解析/AD目标最大差2.28e−13、梯度相对差1.70e−10。Source归一化
  随λs变化另有真实数据核对，见下文。
- 被测代码、输入和以前结果hash保持不变。补充确认另测父结果和代码hash不变；
  父实验未改。所有新实现仅为workspace测试脚本和example检查，库默认实现未变。
- PNG/PDF/SVG、全部555条source CSV和原图NPZ已导出；面板对齐、PDF字号、碰撞
  和逐panel视觉检查通过，见对应figure QA。

## 实验协议

本次只把上轮固定的source λs加入超参数搜索；ρ=5.530494995245579、M1模板和
完整source_scale图继续固定。100×100 source亮度像素仍自由更新；macro、lens light、
2293数据像素、noise、mask、PSF、nsub=4、source bbox、1134个potential系数及802个
core均保持不变。不重跑smooth，不引入新subhalo参数或source rescaling。

C1 cardinal bicubic及固定零外侧节点、JAX Float64、原hard-M3白化参考、source和
potential prior定义均沿用上轮。所有权重候选从同一个冻结source初值和零potential
冷启动，不以新λs重算source初值，也不从邻近候选热启动。**仅释放λs，未释放ρ。**

在log10空间搜索三个强度，顺序为λs、λ4、λ0：

```
log10 λs ∈ [log10(λs_M3)−2, log10(λs_M3)+2]
           = [−5.484901608395098, −1.4849016083950977]
log10 λ4 ∈ [−1.2051919733248866, 7.794808026675113]
log10 λ0 ∈ [ 0.9378228269292088, 6.937822826929208]
λs_M3参考值 = .00032741486396532473
```

Source λs变化会同步进入完整penalty、白化及evidence归一化，而不是仅改变拟合罚项。
完整source–potential耦合endpoint local-GN Gaussian evidence与上轮相同，保留source
积分和归一化；不是exact evidence或完整Hessian Laplace。

L-BFGS-B：maxiter3000、maxfun120000、maxcor30、maxls80、ftol1e−15、gtol1e−8。
验收仍为native success、finite/SPD、GN诊断≤.001。所有拒绝点保留原始状态和
diagnostic evidence，正式log_evidence为空，不参与选优；κ峰不参与选择。

先重测强prior、上轮固定λs最优点、单步法三项最优权重以及M3 λs＋单步lens权重。
DIRECT原版locally_biased=False，名义maxfun160、maxiter50、len_tol.002，显式新拟合
cap220。随后从DIRECT后的最高合格分数点及单步法三项权重各做bounded Powell，
maxfev240、maxiter12、xtol.002、ftol1e−8。增加预算是为了第三个超参数，物理条件不变。

在保留点±.02dex的三维26个邻近点检查；若evidence改善超过.001，再精修。
最后原始候选加5次额外冷启动，共6次全通过、logE范围≤.01、全core最大两两κ差≤1e−4，
才通过重复性门槛。不用更高分复跑覆盖原保留状态。这些是同一数据的技术重复，
不是独立噪声观测或后验样本。有限DIRECT不提供全局最优证明。

## 检验与复现

新增不同source λs的完整小矩阵检查，加既有C1/边界/handoff检查：16 passed，18.64秒。
真实数据同lens权重、不同λs的两个基准点间，logdet prior变化为−3611.6337964200793，
与10000 log(λs之比)完全一致，证明本次source归一化随λs更新。

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python workspace/iteration/run_cubic_evidence_joint.py --output workspace/iteration/cubic_evidence_joint_new
python workspace/iteration/analyze_cubic_evidence_joint.py workspace/iteration/cubic_evidence_joint_new
python workspace/iteration/plot_cubic_evidence_joint.py workspace/iteration/cubic_evidence_joint_new
python workspace/iteration/confirm_cubic_evidence_joint.py workspace/iteration/cubic_evidence_joint_new --output workspace/iteration/cubic_evidence_joint_confirm_new
```

入口：[运行](run_cubic_evidence_joint.py)、[核对](analyze_cubic_evidence_joint.py)、
[出图](plot_cubic_evidence_joint.py)。正式目录为`cubic_evidence_joint_v1/`。

结果：[数值汇总](cubic_evidence_joint_v1/analysis.json)、
[重复认证](cubic_evidence_joint_v1/maximum_evidence/validation.json)、
[场差核对](cubic_evidence_joint_v1/postfit_field_comparison.json)、
[source归一化核对](cubic_evidence_joint_v1/source_normalization_audit.json)、
[拟合与κ图](cubic_evidence_joint_v1/figure/diagnostics.png)、
[额外局部确认](cubic_evidence_joint_confirm_v1/summary.json)。
