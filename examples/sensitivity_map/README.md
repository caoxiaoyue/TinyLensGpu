# Sensitivity map 的 BIC

修改或核对 [model.py 中的 `compare_bic()`](model.py) 时，先按下文确认似然输入和自由参数计数。这里比较 Stage 3 的平滑透镜与 Stage 4 的平滑透镜加 Pseudo-Jaffe 子晕。

## 源边际化似然

设 $d$ 是参与拟合的观测像素，$s$ 是像素化源，$\eta$ 是质量参数，$\lambda$ 和 $\rho$ 是源正则化超参数，$a$ 是 Stage 1 保存的参考弧图。Suyu 框架在源反演层使用 $p(d\mid s,\eta)$。给定源像素时，图像 $\chi^2$ 描述这个高斯似然的残差项。将源像素积分掉得到源反演层的 evidence。[Suyu et al. (2006)，式 (5) 和 (19)](https://arxiv.org/pdf/astro-ph/0601493)

$$
E_{\rm src}(\eta,\lambda,\rho)
= p(d\mid\eta,\lambda,\rho,a)
= \int p(d\mid s,\eta)\,p(s\mid\eta,\lambda,\rho,a)\,\mathrm{d}s .
$$

在质量和正则化参数的外层拟合中，$\log E_{\rm src}$ 就是 log likelihood。**BIC 取各阶段 L-BFGS-B 拟合终点保存的 `log_evidence`。** 本例的 Overlap 分数已包含图像残差、源正则化、行列式比和高斯噪声归一化。`chi2` 用于图像残差诊断。

## Stage 3 与 Stage 4 的 BIC

从 `results.json` 读取 `stages.stage_3.log_evidence` 和 `stages.stage_4.log_evidence`，分别记为 $\ell_3$ 和 $\ell_4$。$N$ 取共用掩膜下的有效原生图像像素数 `np.count_nonzero(~mask)`。$k_i$ 取该阶段 `free_parameters` 的长度。BIC 的似然项与参数惩罚如下。[Schwarz (1978)](https://doi.org/10.1214/aos/1176344136)

$$
\mathrm{BIC}_i=-2\ell_i+k_i\ln N,\qquad
\Delta\mathrm{BIC}_{4-3}=-2(\ell_4-\ell_3)+(k_4-k_3)\ln N .
$$

| 阶段 | 计入 $k_i$ 的参数 | $k_i$ |
|---|---|---:|
| Stage 3 | EPL 的 6 个参数和外剪切的 2 个分量 | 8 |
| Stage 4 | 上述 8 个参数、子晕的 $b_{\rm sub},x_{\rm sub},y_{\rm sub}$，以及重新优化的 $\log\lambda$ | 12 |

Stage 3 固定从 Stage 2 继承的 $\lambda$ 和 $\rho$，Stage 4 继续固定 $\rho$。源像素的贡献已通过边际化进入 $\ell_i$。$\Delta\mathrm{BIC}_{4-3}<0$ 时，BIC 偏向含子晕的 Stage 4。

## 验算

确认 `bic.likelihood` 为 `source_marginalized_overlap_log_evidence`。用保存的 $\ell_3$、$\ell_4$、$N$ 和自由参数数目重算上式，并与 `bic.delta_stage4_minus_stage3` 核对至数值精度。

一次 Float32 模拟的保存值为 $N=4936$、$\ell_3=3435.1113$、$\ell_4=3633.0869$。代入 $-2(3633.0869-3435.1113)+4\ln(4936)$ 得到 $\Delta\mathrm{BIC}_{4-3}\simeq-361.934$。
