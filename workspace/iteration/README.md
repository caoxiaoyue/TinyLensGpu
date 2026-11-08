# J0946 / Jackpot：迭代 potential correction 实验

## 本目录的可复用经验与 C1 实验入口

- [三次 Hermite 插值教学](cubic_hermite_tutorial.md)：从线性插值、端点坡度、
  四个基函数到 C1 拼接与实际 source 实现，附交互式 HTML 教程。
- [JAX＋白化＋L-BFGS 教程](lbfgs_jax_whitening_tutorial.md)：用 Python 伪代码解释
  source–potential 联合目标、白化坐标、自动微分、SciPy 接口及实际低秩实现的对应关系。
- [Source 插值光滑性与 L-BFGS 收敛：经验笔记](lbfgs_source_interpolation_lessons.md)：
  问题机制、受控证据、C1 实现要点、复用检查顺序和结论边界。
- [C1 同权重对照](cubic_source_results.md)：双线性与 C1 的收敛、计时和重复性。
- [固定 M3 λs 的二维 evidence 选优](cubic_evidence_results.md)。
- [释放 λs 的三维 evidence 选优](cubic_evidence_joint_results.md)。
- [未来 TODO：分段重白化 L-BFGS](future_improvements.md)：
  初值较差或优化远离曲率参考时，从当前累计解刷新白化并重启；暂不实现，待验证收益与成本。

以上经验暂存于本实验目录，尚未统一整合到库的经验文档或默认实现。
下文保留最初 `pt.py` 双线性方案的运行说明；C1 测试请使用上述笔记链接的专用脚本。

## 早期双线性实验说明

入口为 `pt.py`。只读取这里已复制的 M3 handoff，不导入或自动执行
`macro_model.py`。最初从 [SLACS J0946 工作目录](../slacs0946_potential_correction/)
逐字节复制了该脚本以及 `macro/` 和 `data/`。
`copied_smooth_manifest.json` 记录了 37 个文件的 SHA256。生成数据及输出被 Git
忽略，Python/Markdown 保留为可审查文件。请保留本地 handoff 才能复现实验。

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python workspace/iteration/pt.py
```

默认输出 `workspace/iteration/potential/`，已有非空目录会被拒绝覆盖。
使用 `--output NEW_DIRECTORY` 创建独立实验。先验证接口与代数可运行：

```bash
python workspace/iteration/pt.py --validate-only --output workspace/iteration/check
```

`--validate-only` 的 `summary.json` 会分别报告代数审计与内层是否合格；
仅进程正常退出不代表已得到合格恢复。默认数值检查 anchor 为 λs 范围中点、
λ4/λ0 上界（强 prior），不缩小 DIRECT 的完整搜索范围；`--validation-log10`
可明确指定别的检查点。Powell 从 anchor/DIRECT 的最高合格分数开始。

算法与保持固定的条件：

- 复用 M3 最大 likelihood 代表点的 EPL+shear、固定透镜光、mask/PSF、nsub=4、
  100×100 source 网格、0.1 arcsec potential 网格。保留所有 2293 个像素（包括
  8 个 noise=1e8 的弱权重像素），1134 个 potential 系数，802 个导数有效节点。
- 拟合保存的 `lens_subtracted_image`；直接推广到 Float64，不重复减光。
  保存的 source adaptive scale 固定，first-order source prior 与原 M3 一致。
- 内层联合优化 signed source 与累积 potential；每次前向重新计算 corrected rays
  及双线性 source 插值。JAX 自动微分提供梯度，SciPy `L-BFGS-B` 不加参数边界。
  不使用手写 LM/GN 更新，也不加入参数化 subhalo。
- 三个正强度同时变化：`λs Rs` 与 `λ4 D4ᵀD4 + λ0 I`。后者覆盖所有 potential
  节点，D4 是与现有代码一致的 pixel-unit 完整四阶差分，没有额外 jitter。
- 默认先求每候选在 p=0 的 exact conditional source MAP，再联合 L-BFGS；
  这是确定的数据驱动初始化，没有搜索顺序依赖。`--source-initialization saved_M3`
  可检查直接以保存 M3 source 为初值的版本。预条件仍用固定 M3 参考点的完整
  GN metric，不能将它称为 conditional source 新起点的精确 GN metric。
- 每候选只构造一次预条件。利用 Nd≪Ns+Np，在数据空间做完整 determinant 和低秩
  白化，不构造 11134×11134 的 joint curvature；保留 source–potential 全部耦合。
- 终点刷新 AD Jacobian，采用 local-GN 高斯近似 evidence：
  `logE = -F - 0.5[logdet(I+Jw Q^-1 Jwᵀ)+N log(2π)+logdet(noise covariance)]`。
  这是非线性问题的近似边际化，区别于精确 evidence / 真 Hessian Laplace。
- 外层在三个 log10 强度上运行 `scipy.optimize.direct`，然后有界 Powell 局部精修。
  默认预算为 DIRECT 120（SciPy 可在一个 division 内超过名义 maxfun）、Powell 80；
  完整记录原生退出原因，不能把预算用完称为全局收敛。经验 bounds 规则与旧单步法
  相同，数值从本次 Float64 响应推导，完整值写入 protocol。

内层资格检查同时要求 SciPy success、有限/SPD 的 evidence，以及
`.5 gᵀ H_GN^-1 g ≤ 1e-3`。最后一项只是预先声明的局部二次模型诊断，
不是实际 objective gap 的上界或 evidence 精度保证。失败保留原始状态与诊断
evidence，但不给正式评分。不得根据图像观感或预期 subhalo 位置挑解。

输出包含每候选日志/状态、最佳合格场与图像、协议/源码快照/校验和、原生外层状态。
最佳点另从冷初值重复，独立 QR 核对 determinant，核对公开 analytical objective/
gradient 和初始白化二次型。单个候选失败不阻止检查其他正则化区域；若 DIRECT
结束后仍无合格解，则显式失败，不导出伪最佳解。

这是实测数据，没有 injected κ 真值。恢复的 δκ 是相对固定 M3 宏模型的修正，
峰值本身不等于 subhalo 检测显著性或质量。M3 的旧 nested logZ 包含位置惩罚且用
block determinants，不能与本实验 logE 做 Bayes factor。

已发现的原模型限制：source 在 bbox 外突然截零，边界非零时使 objective 不连续。
某些超参数下 L-BFGS 会卡在该边界；`diagnose_endpoint.py` 对已保存的失败点做
双侧步长检查及 L-BFGS 重新启动，保留诊断，不改变 source 插值定义或验收阈值。
具体试验结果见后续 `results.md`，接口/数学独立审查见 `review.md`。

本轮已执行的完整探索见 `direct_powell/`：254次拟合/253个不同超参数点，保留候选
δκ峰约0.0437、位置(−0.7″,1.0″)。同参数三次恢复场很接近，但一次严格梯度门槛略
未通过，`summary.json` 因此明确为 `repeat_gate_failed`。外层预算退出，尚不宣称
全域最优或已稳定检测 subhalo。当前默认检查点与另一次独立进程验证均已通过；
所有尝试均在 `results.md` 并列报告，未用后一次通过抹去此前未过。

边界问题的小规模机制验证见 `boundary_probe.md` 与 `check_boundary_stall.py`：
原硬边界可稳定复现梯度未归零的停滞，固定零亮度环在同一小例子中三次均2步收敛。
这只是小例子的诊断结果。

已进一步完成真实 J0946 固定单步 λ 的边界/初值对照，入口为
`run_fixed_boundary.py`，结果为 `fixed_boundary_v2/`，详见
[固定 λ 对照报告](fixed_boundary_results.md)。四种条件各三次，外边缘连续性修复
生效，但12次均未通过原梯度门槛；单步初值可继续降低非线性目标，仍未解决内部
双线性插值折点附近的停滞。κ峰约.1958，全部按诊断候选保存，没有给正式 evidence
评分。此实验使用 `source_boundary.py` 的 workspace renderer，未改变默认生产边界。

2026-11-06 对“解稳定”和“梯度收敛”作了单独复查，见
[双线性解稳定性报告](practical_convergence_results.md) 与
[诊断入口](check_practical_convergence.py)。旧双线性终点续跑的恢复场基本不变，
即使 GN 驻点诊断仍未通过；交叉初值又能找到略低的同模型目标值，说明还存在
精细求解的路径依赖。报告量化了这些差异，并与更换 Hermite 前向模型的差异区分。

进一步加入目标函数与 source/potential/κ 的窗口稳定停止条件，完成 66 次配对拟合，
见 [停止条件耗时对照](stability_timing_results.md) 与
[运行入口](run_stability_timing.py)。此前约 40 倍差距的组在默认实用容差下变为
约 4.9 倍；原判据与新判据的完整状态、监控成本、三档容差及恢复场差异分别保留。
