# SLACS J0946+1006 势校正

两种势校正共用 `macro_model.py` 导出的 M3 handoff。

- [`macro_model.py`](macro_model.py) 从原始 FITS 重新构建 arc mask，并完成 A、B、M0、M1、M2、M3。
- [`pt.py`](pt.py) 在固定 M3 源梯度下执行单步势校正，用 Nautilus 采样三个正则化强度。
- [`pt_iter_lbfgs.py`](pt_iter_lbfgs.py) 用 L-BFGS-B 迭代联合优化源亮度和势校正，用 DIRECT 搜索三个正则化强度。

原始 FITS 从原 J0946 目录复制到本目录 `data/`。单步结果见 [analysis.md](analysis.md)，
联合 QR 复算见 [joint_qr_impact.md](joint_qr_impact.md)，L-BFGS-B 收敛结果见
[lbfgs_regularization_convergence.md](lbfgs_regularization_convergence.md)。

## 运行

从仓库根目录运行宏模型和单步校正。

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python -u workspace/slacs0946_potential_correction/macro_model.py
python -u workspace/slacs0946_potential_correction/pt.py
```

已有本次实验的 manifest 或 checkpoint 时，加 `--resume`；此选项只续跑本目录的新实验，
不会读取原 J0946 目录的建模结果。每约 30 秒更新 progress/checkpoint；JIT 或单次
likelihood 批处理耗时较长时，更新可能延迟。两个脚本不会自动互相启动。
PT 续跑会同时核对 NPZ 校验值和 manifest 中保存的完整 handoff JSON；
宏观质量参数或 source 正则化设置改变时，会在准备 Dense 矩阵前拒绝续跑。
2026-11-02 的联合 QR 修复也更新了 numerical policy：旧版联合 Cholesky
checkpoint 不可直接续跑新评分。重新采样时指定新 `--output` 目录；旧结果保持原样。

```bash
python -u workspace/slacs0946_potential_correction/macro_model.py --resume
python -u workspace/slacs0946_potential_correction/pt.py --output workspace/slacs0946_potential_correction/potential_qr_v2
# 此目录已有新版本 checkpoint 后才使用 --resume：
python -u workspace/slacs0946_potential_correction/pt.py --output workspace/slacs0946_potential_correction/potential_qr_v2 --resume
```

`macro_model.py --prepare-only` 只复制 raw inputs 并记录 manifest，不拟合。
`pt.py --validate-only` 检查矩阵、数值和公开 Dense API 的一致性，不采样。
`pt.py --plot-only` 直接读取保存的数组重新出图；`pt.py --audit-only` 复算保留样本和数值失败候选。

迭代校正使用已有 M3 handoff，新的搜索需要指定新输出目录。

```bash
python -u workspace/slacs0946_potential_correction/pt_iter_lbfgs.py \
  --output workspace/slacs0946_potential_correction/potential_lbfgs_new
```

## 宏观模型

科学设置沿用原流程，包括原有 operator/PCG 像素源 likelihood；本次要求的
Dense 后端用于后续 potential correction。Stage A 是 MGE lens/source light + SIE/shear；
M1 重建非自适应 source template；M2 拟合 source 正则化强度和 adaptive rho；
M3 拟合 EPL/shear，固定 M2 的 source 正则化超参数。

- source 为 100×100，图像 pixel scale 0.05 arcsec，nsub=4；
- source 正则化为 first-order，M3 的 adaptive 权重来自新 M1 template 和新 M2 rho；
- 各采样 stage 使用 300 live points；A 的 n_eff=2000，M1/M2/M3 的 n_eff=600；
- `posterior_*.npz` 保存样本、权重、逐样本 likelihood 和参数顺序；
- M3 handoff 使用**保留样本中 likelihood 最大的质量参数点**，重新求解该点的 source；
  同时记录 posterior medians，避免把中位数和最佳样本混为一谈。

`macro/m3_handoff.npz` 和 `macro/m3_handoff.json` 组成宏模型与势校正之间的接口。
包含 raw/subtracted image、lens light、noise、PSF、mask、source 网格和亮度、
adaptive 权重、template、质量参数和 SHA256。

## 单步 potential correction

固定 M3 的宏观质量模型、source 网格、mask、PSF、nsub；在 M3 rays 上，解析求
同一双线性 source 插值的亮度梯度并冻结。**Source 像素亮度会和 δψ 一起重新求解**，
不是只在固定 source 上拟合 δψ。Adaptive rho 和权重图始终固定，不随新 source 更新。

- Dense、Float32，完整耦合 Gaussian evidence；默认在 observation space 对完整
  covariance 做 Cholesky（使用精确 determinant lemma），不丢弃 source–potential 交叉项；
  `--dense-form joint` 使用先验白化后的增广 QR，通过同一 determinant lemma
  计算完整联合 Gaussian evidence。QR 的列数为观测像素数，保留全部源—势耦合，
  避免一万个源系数对应的巨大参数空间 QR。两者均没有 operator determinant；
- potential grid 间距默认 0.1 arcsec，可通过 `--potential-spacing` 指定；
- 势正则化为 `lambda4 * D4.T @ D4 + lambda0 * I`，第四差分采用 pixel 单位；
- `lambda_s`、`lambda4`、`lambda0` 使用独立有界 log-uniform hyperpriors；
- 默认 lambda_s 的 log10 范围为 M3 值 ±2 dex；lambda0 中心取势响应 data Gram
  对角线的中位数、范围 ±3 dex；lambda4 中心为同一尺度除以 D4 精度正对角线中位数、范围 -3/+6 dex。
  首轮范围为 ±3 dex，但 λ4 后验碰到上界；该轮完整结果保留在 `potential/pilot_data_only/`。
  这是数据相关的经验范围选择，manifest 保存实际边界；也可显式给三个 `--log10-lambda* lo hi`；
- Nautilus：n_live=300，n_eff=1000，n_batch=8；Dense 候选逐个求解以控制显存；
- 对称对角缩放只改善线性代数条件，不改变物理先验或 determinant；不加 jitter，
  不使用 Float64 fallback。数据协方差形式失败时，用同为 Float32 的增广 QR 复算；
  lambda4/lambda0 > 1e4 时以 prior root 的 QR 分解避免平方条件数丢失 lambda0。
  联合 QR 不再显式构造后验 Gram 矩阵：若 Q=L Lᵀ，V=L⁻¹ Mᵀ N⁻¹/²，
  则堆叠 `[V; I]` 的 QR 给出 C=I+VᵀV 的根；log det H−log det Q=log det C。
  解与 determinant ratio 来自一致的根，验收通过原始响应和有限差分算子重算残差。
  两种形式均失败的候选仍逐项记录，不能解释为物理排除。

采样前用一组 probe 检查，并在其中三个较好点与公开 Dense fitter 对比。输出
`validation.json`；采样记录 `evaluations.jsonl`，保留 Float32 QR 复算次数和未解决失败次数。数值失败若进入 posterior
相关区域，必须重新审查结果，不能把失败当成物理排除。

`potential/posterior.npz` 和 `posterior_summary.json` 保存加权后验；
`maximum_evidence/` 保存采样中最大 conditional evidence 点的重建；
`posterior_median/` 保存三个边缘后验中位数处的条件重建。
两者均包含原始 signed source/δψ、δα、δκ、线性和非线性重渲染及归一化残差。
`result.json` 记录实际求解方式、determinant 方法和 Float32 回退状态；
`requested_dense_formulation` 另行保留用户请求的方式。
Derivative map 在 support-only 节点未定义，保存为 NaN，不外插填满。

这是固定宏观模型、固定 source 梯度条件下的重建实验。M3 的 nested evidence
包含位置惩罚和 operator determinant，不能直接作为 PT 的模型比较基线。
本实验的 κ 峰不自动等同于 subhalo 显著性或质量测量；应同时检查非线性重渲染残差。

## 验证

核心测试和 example 测试分开进程运行：

```bash
cd tests
pytest test_potential_correction_dense.py test_potential_correction_precision.py -q
cd ../examples/tests
pytest test_j0946_single_step_workflow.py -q
```

`test_strong_d4_joint_and_fallback_complete_evidence` 直接运行实际缓存求解器，
在五节点 D4 的未观测及源—势耦合例上核对完整 evidence、两个 log determinant
和系数解；解析参考独立使用 Float64 秩一逆。强耦合例若触及 Float32 的实际
方程残差下限，仍按原门限返回失败，测试另核对完整未门控评分。

对旧 J0946 campaign 的只读复算：

```bash
python -u workspace/slacs0946_potential_correction/audit_joint_qr.py \
  --output workspace/slacs0946_potential_correction/joint_qr_replay
python -u workspace/slacs0946_potential_correction/audit_joint_qr.py \
  --output workspace/slacs0946_potential_correction/joint_qr_replay --phase reference
```

首次复算使用新输出目录。已有进度可用 `--resume`，需匹配 manifest 中的 driver 和输入校验值。
第二步在同一冻结 Float32
响应/网格上用独立 SciPy Float64 线性代数检查代表点；不调用 Float64 likelihood
作为生产 fallback，也不改变原 sampler、后验、图像或失败记录。

输入、checkpoint、日志和大数组留在本目录且遵循 gitignore；建模入口、复算脚本和本文档可追踪。
