# J0946 单步 potential correction：本次从头运行的结果

日期：2026-11-01。两个入口为 `macro_model.py` 和 `pt.py`。
本次只复制原始 image/noise/PSF FITS，重新运行 A、B、M0、M1、M2、M3，
没有使用原目录的 posterior、source reconstruction、feature mask 或 M4/M5 结果。

2026-11-02 的联合 QR 修复及影响见 [joint_qr_impact.md](joint_qr_impact.md)。
本文以下数值保留原版实验记录。当前驱动升级了 numerical policy，不能将旧
checkpoint 直接续跑到新评分；旧 campaign 的复算使用 `audit_joint_qr.py`，
新采样使用独立的 `--output` 目录。

## 设置与交接

宏观拟合沿用原 pipeline 的 operator/PCG likelihood。M3 完成后，选择保留样本中
likelihood 最大的参数点，再求解该点的 source，导出 `macro/m3_handoff.npz` 和 JSON。
最大 likelihood 样本的 theta_E=1.40248382 arcsec、gamma=2.00401282；
这与 posterior median 参数点是不同的代表选择，完整两套参数都在交接 JSON 中。

PT 固定这个 EPL+shear、source 网格、M1 adaptive template、M2 的 rho=5.530495、
其最终权重图、PSF、mask 和 nsub。Source 为 100×100、nsub=4，观测保留 2293 个像素。
势网格间距 0.1 arcsec，1134 个拟合节点，802 个 derivative-valid core 节点。
Source 梯度来自同一个 M3 source 的双线性插值，并在整个单步求解中冻结。
Source 像素和势修正仍然一起重新求解，均允许正负值。

PT 使用 Dense、Float32。默认完整数据协方差 Cholesky 与完整联合矩阵 Cholesky
通过 Gaussian determinant lemma 等价，保留全部交叉项。数据形式未通过数值检查时，
使用同为 Float32 的完整联合矩阵复算。没有 Float64 likelihood、jitter 或 operator determinant。
较大 lambda4/lambda0 比值下，势 prior root 使用 QR，避免平方条件数丢失 lambda0。

## 三个正则化强度

先验为三个有界 log-uniform。实际 log10 边界按 lambda_s、lambda4、lambda0 排列：

- lambda_s：[-5.4849016084, -1.4849016084]
- lambda4：[-1.3235741427, 7.6764258573]
- lambda0：[0.8194406575, 6.8194406575]

这些是数据相关的经验尺度选择，不能据此把 hyperprior-integrated evidence 当作
经过校准的模型比较结果。首轮 lambda4 上界为 4.6764258573，后验碰到边界；
首轮所有产物留在 `potential/pilot_data_only/`，最终实验扩大其上界并重新开始采样。
最终 lambda4 在最外侧 2% 先验区间中的 posterior weight 仅约 1.87e-7。

Nautilus 使用 n_live=300、目标 n_eff=1000、seed=947；最终调用 6944 次，
保留 1144 个 weighted posterior points，n_eff=1003.586。

| 强度 | 后验中位数 | 16%–84% 区间 |
| --- | ---: | ---: |
| lambda_s | 2.26494e-4 | [2.01339e-4, 2.53116e-4] |
| lambda4 | 7.17533e4 | [3.51249e4, 1.49010e5] |
| lambda0 | 4.51633e4 | [2.36338e4, 7.87787e4] |

采样中最大 conditional evidence 点的强度为 lambda_s=2.29454e-4、
lambda4=5.90631e4、lambda0=5.08403e4。`maximum_evidence/` 和 `posterior_median/`
分别保存这两类条件重建；后验中位数处的 map 不是 map posterior mean。

## 重建结果

最大 evidence map 的正 delta-kappa 峰为 **0.21570**，位于 **(-0.6, 1.1) arcsec**；
后验中位数 map 的峰为 **0.20939**，位于同一 coarse-grid 节点。
左上弧附近的局部正势质量信号在两种强度选择下都存在；其它弧段也存在正负结构。
这里报告的是离散 delta-kappa 峰的位置，不能直接当作参数化 subhalo 的中心或质量测量。

| 模型/图像 | chi2（相同 2293 个观测像素） | residual/noise RMS |
| --- | ---: | ---: |
| 固定起点 M3 的 source 预测 | 1312.24 | 0.75649 |
| PT 最大 evidence 点，线性预测 | 938.51 | 0.63976 |
| 同一 PT 解，非线性重渲染 | 1009.81 | 0.66362 |

非线性重渲染相对线性预测有所退化，说明单步线性化有可见影响；但残差仍小于起点 M3。
这些改善包含 source 和正则化强度的变化，不能单独归因于 subhalo。
本次没有进行 source-only 对照、宏观质量不确定性的边缘化、网格/梯度鲁棒性校准，
因此不报告 subhalo detection significance 或质量区间。

图像诊断、source/收敛率图和 corner 图均有 PNG、PDF、SVG；
原始 source、delta-psi、delta-alpha、delta-kappa、线性/非线性残差均保存在对应 reconstruction.npz。
只显示实际观测选择范围；support-only 的导数保持 NaN，不填补为观测约束。

## Float32 数值检查及边界

本次 793/6944 次候选触发 Float32 联合矩阵复算；其中 102 次仍未通过 1e-5 残差门限，
约占 **1.47%**，这些候选在采样中返回 -inf。这是数值失败，不是物理排除。
全部失败点及复算细节保存在 `failed_candidate_audit.json`。

另外对所有 1144 个 retained points 用另一种完整 Dense Float32 形式复算：
1140 个通过门限，余下 4 个点的原 posterior weight 在保存的浮点权重中为零。
在有效支持集上，joint-minus-sampled log evidence 的加权均值为 -0.000463、
加权 RMS 为 0.003356，最大绝对差为 0.27246。
基于相同支持集的重要性重加权 TV 为约 0.131%，三个中位数的变化很小。
这是代数形式的一致性检查，不是 Float64 检验，也不是独立 sampler 重跑。

最大 evidence 点的公开 Dense fitter 与保存结果的 log evidence 差为 0.00342，
source 相对 L2 差约 0.0100%，delta-kappa 相对 L2 差约 0.0219%。
后验中位数处也通过同样的公开 API 检查。

失败点到已采样后验中心的最小 Mahalanobis 距离约 6.72；
失败解未加数值门限的最高形式分数为 6291.45，低于保留样本最大值 6304.80。
这些仅支持已采样中心区域的实用数值一致性，不能证明失败或未采样先验体积的覆盖。
若用于正式统计结论，应进一步解决失败点并进行独立精度与采样检查。

详见 `potential/numerical_audit.json`、`joint_posterior_checks.json` 和 `validation.json`。
不将 M3 的 operator nested evidence 与这里的 conditional evidence 相减。

## 运行与检查

读取原结果，或将 QR 复算写入新目录。

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python workspace/slacs0946_potential_correction/macro_model.py --resume
python workspace/slacs0946_potential_correction/pt.py --plot-only
python workspace/slacs0946_potential_correction/audit_joint_qr.py \
  --output workspace/slacs0946_potential_correction/joint_qr_replay
```

23 项核心测试通过，4 项流程/代数/QR/fallback 测试通过；Black、Ruff 与 diff whitespace 检查通过。
完整环境与源码 hash 记录在 `run_environment.json`，各阶段 posterior 和采样 checkpoint 均已保存。
