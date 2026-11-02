# J0946 联合 QR 修复与影响复算

日期：2026-11-02。复算对象是 [2026-11-01 的固定 M3 实验](analysis.md)，
原始数据、M3 交接、采样 checkpoint、后验及重建均保留。
完整复算数组、逐点评分和输入校验记录在 `joint_qr_impact_20261102/`。

## 修复的数值问题

旧版联合求解用 QR 计算势先验的根及 log det Q，但仍显式构造
H=MᵀN⁻¹M+Q，再用 Float32 Cholesky 求解和计算 log det H。
强 D4 下，H 的弱方向和行列式可能失真。旧验收只检查这个舍入矩阵的残差，
无法证明完整 Gaussian evidence 准确。

五节点 D4、lambda_s=1e-3、lambda4=4e6、lambda0=10 的 CPU 解析例中，
旧联合评分比解析值高约 0.59049，仍通过残差门限。无势观测时，
势先验的积分本应恰好为一，任何这样的评分偏移都是数值误差。

修复保留默认的数据协方差 Cholesky，替换 `--dense-form joint` 和
默认路径的 joint fallback。若 Q=L Lᵀ、y=N⁻¹/²d、V=L⁻¹MᵀN⁻¹/²，
则 C=I+VᵀV，且 det H / det Q = det C。对增广根 `[V; I]` 做 QR，
直接得到完整 determinant ratio，避免再次形成后验 Gram 矩阵。
全部源—势耦合均保留，没有分块 determinant 近似。

列缩放后令 `[V; I] D⁻¹ = U R`。系数解使用
`z=R⁻ᵀ D⁻¹y`、`x=L⁻ᵀ U_top z`，直接利用正交因子恢复，
避免先求 dual 再乘 V 时，近正交项相消的误差被弱先验逆放大。
QR 的列数为 2293 个观测像素，而不是 11134 个联合系数。

验收通过原始响应和有限差分先验重新计算方程残差。
仍为纯 Float32，最高矩阵乘法精度，`solve_rtol=1e-5`，没有 jitter、
Float64 生产 fallback 或残差门限放宽。
数值方法版本为 `float32_data_with_prior_whitened_joint_QR_v2`；
旧 checkpoint 会在准备 Dense 矩阵前被拒绝，防止旧评分混入新采样。

## 同一批后验点的复算

保留原 source=100×100、nsub=4、2293 个观测像素、1134 个势节点、
802 个 derivative-valid 节点、0.1 arcsec 势网格及原 M3 adaptive source 权重。
全部 1144 个保存样本均复算。QR 有 3 个无效评分，原记录另有一个
零权重的非有限评分；共同有限集为 1140 点，覆盖全部非零原后验权重。
无效点没有被隐式解释成物理排除。

| 相对原保存评分 | 新默认路径 | 全部点强制 QR |
| --- | ---: | ---: |
| 原后验权重中未通过的质量 | 0 | 0 |
| 加权平均 delta log evidence | -2.30e-5 | +0.0040465 |
| 加权 RMS delta log evidence | 0.0003225 | 0.0040857 |
| 重要性重加权总变差 | 0.01058% | 0.02266% |
| 重加权 Kish 有效样本数 | 1003.60 | 1003.59 |

强制 QR 的三个强度中位数相对变化分别为：lambda_s -0.00115%、
lambda4 +0.00752%、lambda0 -0.00310%。默认路径的最大中位数变化约为
0.000774%。这衡量的是**同一支持集上的评分变化**。

尾部的最大有限评分变化仍为 0.421875；该点原权重约为 2.59e-40。
加权结果稳定并不等同于逐点 evidence 都达到千分之一精度。

## 同一强度处的条件重建

以下始终使用原采样最优强度和原三个边缘中位数，单独检查求解器变化。
相比原保存图像，支持节点和 derivative-valid 域相同，没有裁剪或仿射对齐。

| 指标 | 原最优强度点的 QR 解 | 原后验中位数处的 QR 解 |
| --- | ---: | ---: |
| log evidence | 6304.80615 | 6304.75098 |
| 相对原评分变化 | +0.003906 | +0.003906 |
| 原 peak delta-kappa | 0.215700 | 0.209389 |
| 新 peak delta-kappa | 0.215706 | 0.209401 |
| 峰位置（arcsec） | (-0.6, 1.1) | (-0.6, 1.1) |
| Source 相对 L2 变化 | 0.00412% | 0.00432% |
| 势系数相对 L2 变化 | 0.01499% | 0.01559% |
| 有效核心 delta-kappa 相对 L2 变化 | 0.01409% | 0.01547% |
| 原始算子计算的相对方程残差 | 1.64e-7 | 1.65e-7 |

结果支持原中心重建的稳定性；不将该离散峰解释为 subhalo 显著性或质量测量。

## 原评估候选与失败点

对原全部 6944 次候选重新运行新默认路径，包含所有原 fallback 和失败点：

- 原 793 次 fallback，本次 801 次；边界附近的 Float32 调用存在微小运行差异。
- 原 102 个失败候选恢复 10 个，另有 1 个原有效候选被原始算子残差检查拒绝；
  剩余 93 个失败候选仍保留为数值失败。
- 最高恢复评分为 6287.2363，比原保留样本最高分 6304.8022 低约 17.57。
- 新拒绝的原有效候选旧分为 6251.5283，位于已采样后验中心之外。
- 共同有效候选的最大评分变化为 1.20996；发生在低评分、强先验尾部。

这些结果检查了实际评估过的候选，不能证明未采样先验体积的覆盖。

## 独立 Float64 算术对照

使用独立 SciPy Float64 Gaussian marginalization：在同一保存的 Float32
响应、噪声、源先验 stencil 和势差分上，源块用稀疏 LU，势先验和数据协方差
用 Float64 Cholesky。强度使用 GPU 实际计算的 Float32 数值，避免跨后端
exp10 舍入差异。该对照不重建 Float64 几何，也不作为生产 fallback。

| 点 | QR minus Float64 log evidence |
| --- | ---: |
| 原最大评分点 | +0.00458 |
| 原后验中位数 | +0.00486 |
| QR 重加权中位数 | +0.00441 |
| 最高评分的原 fallback 候选 | +0.00443 |
| 最高恢复评分候选 | +0.04497 |
| 共同有效候选中变化最大者 | +0.00447 |

在原最优点和中位数处，source 相对 Float64 的 L2 差约为 0.00256%，
势系数差约为 0.00060% / 0.00089%。Float64 的两种等价 quadratic/evidence
表达式相差约 1e-12–2e-10。
强先验尾部仍有约 0.045 的评分偏差及有效物理解被 Float32 残差门限拒绝的情况；
本修复不建立宽先验范围上的统一 Float32 精度保证。

## 从零重跑同一采样配置

新运行保存在 `potential_qr_v2/`。固定同一 M3 交接、完整原先验边界、
source 权重、seed=947、n_live=300、n_eff=1000、n_batch=8，
从零启动 Nautilus，没有恢复或改写原 checkpoint。
本次完成 6968 次评估，保留 1136 个加权样本，实际 n_eff=1003.00；
793 次 QR fallback，72 次仍失败，约占 1.03%。

| 强度 | 原中位数 [16%,84%] | 新运行中位数 [16%,84%] |
| --- | --- | --- |
| lambda_s | 2.26494e-4 [2.01339e-4,2.53116e-4] | 2.26623e-4 [2.02615e-4,2.51715e-4] |
| lambda4 | 71753 [35125,149010] | 67133 [33038,159868] |
| lambda0 | 45163 [23634,78779] | 47339 [23175,81938] |

新旧中位数分别变化 +0.057%、-6.44%、+4.82%，68% 区间大幅重叠。
从零重跑会改变采样轨迹，不能将这些百分之几的差异全部归于评分修复；
前述同点重要性重加权是隔离评分变化的直接对照。
单次重跑不足以独立估计所有采样误差。

新运行的最大评分为 6304.80566；在其最优强度和新中位数处，
delta-kappa 峰分别为 0.213663 和 0.211201，位置均为 (-0.6,1.1) arcsec。
相比原运行的 0.215700 / 0.209389，峰幅变化约 -0.94% / +0.87%，
同时包含代表强度点的变化。新中位数解的线性/非线性 residual RMS
分别为 0.639984 / 0.663315，原值为 0.640254 / 0.663406。

新旧 hyperprior-integrated log evidence 为 6296.29276 / 6296.33878，
差约 -0.0460。这是相同经验先验边界下的数值/采样对照，
不作 subhalo detection significance 或经过校准的模型比较解释。

## 测试、性能与复现

旧实现的 8 个完整 evidence 回归用例全部失败；修复后通过。
用例覆盖 joint、强制 fallback、无势观测与源—势耦合，并核对完整 evidence、
两个 log determinant 和系数解。极硬耦合例保留严格残差门限，并另核对未门控评分。
新增旧 numerical policy 拒绝测试。

23 项核心测试在 CPU 通过；43 项 example 测试在 GPU 通过。
其中强 D4/续跑保护的 9 项也单独在 CPU 通过。测试进程分开运行。
在 RTX 5080 上，代表点单次 warm QR 约为 0.0475 秒，默认路径约为
0.0043 秒；这些是复算前的同步单次测量，未作为多进程 benchmark 结论。

从仓库根目录、`tinylens_gpu` 环境运行：

```bash
python -u workspace/single_step/audit_joint_qr.py \
  --output workspace/single_step/joint_qr_impact_20261102
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
python -u workspace/single_step/audit_joint_qr.py \
  --output workspace/single_step/joint_qr_impact_20261102 --phase reference
```

已有复算使用 `--resume`，或选择新目录。数值记录包括 `posterior_impact.json`、
`candidate_impact.json`、`map_impact.json`、`float64_reference.json`、
所有逐点评分及新旧输入 hash。12 个原输入/结果文件的 SHA256 复核保持一致。

可追踪的紧凑数值记录为
[summary.json](../../.scratch/potential-correction/research/j0946-joint-qr-20261102/summary.json)。
完整新采样的后验、checkpoint、原始重建和 PNG/PDF/SVG 图像保存在
`potential_qr_v2/`，没有覆盖 `potential/`。
