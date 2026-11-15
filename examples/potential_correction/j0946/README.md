# SLACS J0946+1006 势校正

两种势校正共用 `macro_model.py` 导出的 M3 handoff，本目录是端到端自包含工作流。

```text
data/{image,noise,psf}.fits
        |
        v
macro_model.py  -->  output/macro/  (m3_handoff.npz + m3_handoff.json)
                          |                    |
                          v                    v
                 pt.py (单步, Nautilus)   pt_iter_lbfgs.py (DE + L-BFGS-B)
                          |                    |
                 output/single_step/     output/lbfgs/
```

- [`macro_model.py`](macro_model.py) 从原始 FITS 重新构建 arc mask，并完成 A、B、M0、M1、M2、M3。
- [`pt.py`](pt.py) 在固定 M3 源梯度下执行单步势校正，用 Nautilus 采样三个正则化强度。
- [`pt_iter_lbfgs.py`](pt_iter_lbfgs.py) 用 Float64 dense L-BFGS-B 联合拟合源亮度和势校正，用 SciPy 差分进化搜索三个正则化强度。
- [`run_pipeline.py`](run_pipeline.py) 把上述三个阶段放在相互隔离的子进程中依次运行，
  父进程不导入 JAX 拟合代码，各子进程的精度配置互不干扰。

本目录只保留从原始 FITS 到宏模型与势校正的当前流程；验证使用可重新生成的小型输入，不要求历史实验归档。

## 数据

`data/` 本地保存 `image.fits`、`noise.fits`、`psf.fits` 三份原始输入
（已 gitignore，不随仓库分发）。全新检出
需通过 `--data-dir` 指向已有数据目录，或用 `macro_model.py --input-dir`
复制原始输入。输入的 SHA256 在运行准备阶段写入
`output/macro/manifest.json` 的 `inputs_sha256`。`feature_mask.fits`
不随输入分发：Stage B 每次都从原始数据重新计算。

`--data-dir` 指定本次运行使用的本地原始数据目录（默认脚本旁 `data/`）；
`--input-dir` 指定一个外部原始 FITS 目录，首次运行时把三份输入按字节
复制进 `--data-dir`（已存在的本地副本若字节不同会拒绝运行，绝不覆盖）。
不带 `--input-dir` 时，`--data-dir` 本身必须已含三份输入。

## 运行整条流水线

从仓库根目录运行；`--correction both` 的完整运行开销很大（M3 采样 + 两个校正分支）。

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python -u examples/potential_correction/j0946/run_pipeline.py
```

默认读取 `j0946/data/`，写入 `j0946/output/{macro,single_step,lbfgs}`。
`run_pipeline.py` 依次调用 `macro_model.py`、`pt.py`、`pt_iter_lbfgs.py`
子进程，子进程使用各自不变的默认参数；宏模型完成且 handoff 校验通过后
才启动校正分支。已完成的 M3 handoff 可复用：

```bash
python -u examples/potential_correction/j0946/run_pipeline.py \
  --macro-dir examples/potential_correction/j0946/output/macro \
  --correction both \
  --output-root examples/potential_correction/j0946/output/reused_run
```

`reused_run` 仅为示例名，选择一个未使用的目录。输入准备可单独检查：

```bash
python -u examples/potential_correction/j0946/run_pipeline.py \
  --prepare-only \
  --output-root examples/potential_correction/j0946/output/preflight
```

之后对 `output/preflight/macro` 使用 `--resume-macro` 续跑，
或选择新的 `--output-root` 正式运行。

复用时流水线先校验 handoff 的完整性和 SHA256，再运行校正；
`--macro-dir` 不能与 `--resume-macro`、`--prepare-only` 或
`--source-interpolation` 同用。`--prepare-only` 只做宏观输入准备
（复制原始输入并写出 manifest，不拟合），之后可对新输出目录正式运行，
或对同一 macro 目录使用 `--resume-macro` 续跑。校正输出目录必须全新；
DE 搜索没有 resume。各分支可用 `--correction single-step|lbfgs|both` 选择。

## 单独运行各阶段

`macro_model.py` 和 `pt.py` 的已有 manifest/checkpoint 可加 `--resume` 续跑。
每约 30 秒更新 progress/checkpoint；JIT 或单次
likelihood 批处理耗时较长时，更新可能延迟。两个脚本不会自动互相启动。
PT 续跑会同时核对 NPZ 校验值和 manifest 中保存的完整 handoff JSON；
宏观质量参数或 source 正则化设置改变时，会在准备 Dense 矩阵前拒绝续跑。
续跑必须匹配 handoff、求解策略和采样设置；不兼容的 checkpoint 需要使用新输出目录。

```bash
python -u examples/potential_correction/j0946/macro_model.py --resume
python -u examples/potential_correction/j0946/pt.py \
  --output examples/potential_correction/j0946/output/single_step_v2
# 此目录已有新版本 checkpoint 后才使用 --resume：
python -u examples/potential_correction/j0946/pt.py \
  --output examples/potential_correction/j0946/output/single_step_v2 --resume
```

`macro_model.py --prepare-only` 只复制 raw inputs 并记录 manifest，不拟合。
`pt.py --validate-only` 检查矩阵、数值和公开 Dense API 的一致性，不采样。
`pt.py --plot-only` 直接读取保存的数组重新出图；`pt.py --audit-only` 复算保留样本和数值失败候选。

迭代校正使用已有 M3 handoff，新的搜索需要指定新输出目录。

```bash
python -u examples/potential_correction/j0946/pt_iter_lbfgs.py \
  --output examples/potential_correction/j0946/output/lbfgs_v2
```

差分进化默认使用 24 个种群成员和最多 10 代。用 `--de-popsize` 调整种群倍数，
用 `--de-maxiter` 调整代数。添加 `--polish` 可启用 SciPy 的局部精修。

L-BFGS-B 默认使用 `--gn-stationarity-tol 10000`，这是宽松的有限上限，
不代表已通过严格的驻点检查。结果记录实际 GN 统计量、优化器退出状态和
有限性；可显式设置更严格的门槛。

## 宏观模型

科学设置沿用原流程，包括原有 operator/PCG 像素源 likelihood；本次要求的
Dense 后端用于后续 potential correction。Stage A 是 MGE lens/source light + SIE/shear；
M1 重建非自适应 source template；M2 拟合 source 正则化强度和 adaptive rho；
M3 拟合 EPL/shear，固定 M2 的 source 正则化超参数。

- source 为 100×100，图像 pixel scale 0.05 arcsec，nsub=4；
- source 正则化为 first-order，M3 的 adaptive 权重来自新 M1 template 和新 M2 rho；
  导出的 source scale 为所保留 M3 样本的原生 reference 投影
  （`selected_M3_native_reference_projection`）；
- M1、M3 采样使用 n_live=300，M2 使用 n_live=250；A 的 n_eff=2000，M1/M2/M3 的 n_eff=600；
- `posterior_*.npz` 保存样本、权重、逐样本 likelihood 和参数顺序；
- M3 handoff 使用**保留样本中 likelihood 最大的质量参数点**，重新求解该点的 source；
  同时记录 posterior medians，避免把中位数和最佳样本混为一谈。

`output/macro/m3_handoff.npz` 和 `output/macro/m3_handoff.json` 组成宏模型与势校正之间的接口。
包含 raw/subtracted image、lens light、noise、PSF、mask、source 网格和亮度、
adaptive 权重、template、质量参数和 SHA256。

## 单步 potential correction

固定 M3 的宏观质量模型、source 网格、mask、PSF、nsub；在 M3 rays 上，解析求
handoff 所选 source 插值（默认 bilinear，若 handoff 导出 cubic Hermite 则为
Hermite）的亮度梯度并冻结。**Source 像素亮度会和 δψ 一起重新求解**，
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

`output/single_step/posterior.npz` 和 `posterior_summary.json` 保存加权后验；
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
pytest test_j0946_single_step_workflow.py test_j0946_example_pipeline.py -q
```

`test_strong_d4_joint_and_fallback_complete_evidence` 直接运行实际缓存求解器，
在五节点 D4 的未观测及源—势耦合例上核对完整 evidence、两个 log determinant
和系数解；解析参考独立使用 Float64 秩一逆。强耦合例若触及 Float32 的实际
方程残差下限，仍按原门限返回失败，测试另核对完整未门控评分。

本工作流依赖 `tinylens_gpu` 环境（JAX、nautilus、scipy、astropy）。
完整默认模型建议使用兼容的 GPU 环境；CPU 可用于小规模检查，默认模型的
CPU 可行性未在本次迁移中验证。输入、checkpoint、日志和大数组写入
`data/` 与 `output/`（遵循 gitignore）；建模入口、复算脚本和本文档可追踪。
限制：势校正均为固定 M3 宏观模型的条件重建，κ 峰不自动等同于 subhalo
显著性；`pt.py` 的 Nautilus 采样与 DE 全局搜索均包含随机搜索/采样步骤；
已固定随机种子，跨环境复现仍需保留版本与 manifest。
