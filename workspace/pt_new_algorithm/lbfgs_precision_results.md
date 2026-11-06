# L-BFGS potential correction：JAX float32 / float64 实测

2026-11-03，标准 RTX 5080，tinylens_gpu 环境。

**有扰动标准 Mock 的 warm 完整恢复从 float64 的 0.988 秒降到 float32 的 0.223 秒，观察到约 4.42 倍提速。32 位恢复的主形态、位置和峰值很接近，但没有达到此前相同的目标函数误差容差；额外 control 重复测试还出现一次初始白化矩阵分解失败。因此不能把这个数字解释成“相同精度和稳定性下快 4.42 倍”，也不建议直接改成纯 32 位默认。**

脚本：[benchmark_lbfgs_precision.py](benchmark_lbfgs_precision.py)。原始记录：[lbfgs_precision_v1](lbfgs_precision_v1/summary.json)。独立核查：[lbfgs_precision_review.md](lbfgs_precision_review.md)。没有修改库的求解器、先验或默认精度。

## 对比范围

沿用最初约 1 秒计时的 200×200、seed=17 科学 Mock：22016 个有效像素，2401 个源变量，956 个势变量，724 个有效 core 节点；源 49×49，势间距 0.075 arcsec，生成和反演 nsub=2。固定同一个宏模型、PSF、mask 和正则化，原 penalty 为数据项加 `0.5 * x.T @ Q @ x`，势四阶差分与所有节点幅度项均保持原值。

两种精度都从**同一个 truth-node 源初值、零势修正**开始，以对齐此前 1 秒的计时，而不是使用最近十种子稳定性实验的观测 source-only 初值。该对比不代表真实未知源初始化的额外计时。Mock 统一由 64 位生成，数据和初值逐项 cast；保留同一个势网格整数拓扑，只改变浮点精度，不重新选节点或改变物理先验。canonical 输入和拓扑 hash 在两进程间核对一致。浮点射线、插值和前向函数仍会有预期舍入差异。

算法保持 JAX 自动微分、初始 `J.T W J + Q` 白化、SciPy L-BFGS-B；每次重新计算数值矩阵和 Cholesky，不使用矩阵磁盘缓存或上次最优解。选项完全相同：`maxiter=3000, maxfun=120000, maxcor=30, maxls=80, gtol=1e-8, ftol=1e-15`，没有 bounds、jitter、额外 ridge 或逐步质量片重标定。

float32 worker 的初值、数据、噪声、射线、源轴、插值权重、forward、prior、AD 梯度、H0、Cholesky 和白化变量均实际为 float32，PSF FFT 为 complex64；目标/梯度 JAXPR 中没有 f64/c128。float64 worker 对应 float64/complex128。两者均使用 `default_matmul_precision="highest"`。**SciPy 的 CPU 优化器控制和传回的 host 梯度数组仍是 float64；这里测试的是 JAX 计算用 32 位，不是整个程序所有数组都用 32 位。**

环境：Python 3.11.14，JAX 0.7.1，SciPy 1.17.1，NumPy 2.4.6；GPU NVIDIA GeForce RTX 5080。串行新进程运行，CPU BLAS 线程数限制为 1，没有并发 GPU 拟合。

## 实测耗时

每种精度先做一次首次运行，再从相同初值做 3 次 warm 恢复。计时包含**初始 AD 曲率、白化分解、L-BFGS、最终 AD 梯度诊断和图像/源/势原始场产品**；同步等待 GPU 完成。首次运行包含这些操作的首次 JIT；warm 复用编译函数，但重新计算数值矩阵和分解。

计时不含 Python/CUDA 启动、Mock 生成、几何装配、独立 64 位外验、文件输出、绘图和 evidence。这与此前约 1 秒的计时口径一致，不是从启动 Python 到落盘的总用时。

| 有扰动 Mock | float64 | float32 |
| --- | ---: | ---: |
| 首次完整恢复，含首次 JIT | 5.448 s | 4.375 s |
| warm 第 1 次 | 0.970 s | 0.232 s |
| warm 第 2 次 | 1.025 s | 0.223 s |
| warm 第 3 次 | 0.988 s | 0.221 s |
| warm 中位数 | **0.988 s** | **0.223 s** |
| warm L-BFGS 接受步数 | 190–205 | 55–59 |

观察到的 warm 比值为 4.42，时间减少约 77.4%。首次运行只快约 1.25 倍，因为编译占比大。单独的前置 Mock/几何准备记录为 3.78 / 4.50 秒，其中 32 位 worker 也先准备共同的 64 位参考；这不是优化器计算的加速比，不能把它当作高效纯 32 位数据生成实现。

有扰动 warm 各阶段的中位数：

| 阶段 | float64 | float32 |
| --- | ---: | ---: |
| 初始 AD 曲率矩阵 | 0.4357 s | 0.1105 s |
| Cholesky | 0.0253 s | 0.0052 s |
| 梯度准备 | 0.0084 s | 0.0045 s |
| L-BFGS 优化循环 | 0.3678 s | 0.0515 s |
| 最终梯度诊断 | 0.1360 s | 0.0382 s |
| 重建场产品 | 0.0120 s | 0.0088 s |

各列是分阶段独立中位数，其和不必恰等于总耗时中位数。初始曲率同样工作量约快 3.94 倍，说明确有计算提速；优化循环的较大差异同时包含了 32 位提前停止、步数更少的影响。

## 恢复质量

优化后将每个 endpoint 转回**原始 64 位 renderer/data/prior**，重新计算目标及图像、源和 core κ，外验不参与停止或计时。仍使用此前工程容差：目标超额≤1e-3，图像 RMS 差≤1e-3 倍 noise，源相对 L2 差≤0.005，κ RMS 差≤1e-5+0.01×参考 κ RMS。原生停止状态、旧梯度门和这些一致性检查分开报告，没有因 32 位表现而抬高阈值。

四次有扰动 float32 均由 SciPy 原生正常停止，参数和有效场有限；峰位置均为 **(0.75, 0.075) arcsec**，与 float64 完全一致。float64 峰约 0.303576，float32 约 0.303159–0.303328，峰值差约 **0.08%–0.14%**。

| 与同数据 float64 解的差异，含 cold 和 warm | float32 范围 | 既有一致性检查 |
| --- | ---: | --- |
| 64 位重算目标超额 | 0.001950–0.006770 | **未达到 1e-3** |
| 图像差 RMS / noise | 0.000258–0.000542 | 达到 |
| 源相对 L2 差 | 0.0255%–0.0586% | 达到 |
| core κ 差 RMS | 0.000153–0.000276 | 达到，限值约 0.000349 |

float32 的图像残差 RMS/noise 约 1.00184，与 float64 的 1.00184 接近。主形态和场差都很小，不能仅凭旧梯度门未通过就称为不物理；但也不能称其达到原 64 位优化精度。此处真实注入离散峰约 0.631435，约 0.3036 的恢复峰包含原有正则化压低，并不代表无偏质量恢复。

F 约为 11735 时，float32 相邻浮点间隔是 0.0009765625。原 `ftol=1e-15` 无法让单精度目标辨别任意小的下降；本批 32 位均按相对目标变化停止，步数明显少于 64 位。这与目标量化导致较早停止相符，不能仅凭 `success=True` 认定精度等同。固定相同 `ftol` 也不等于固定了实际达到的数值精度。

## 额外 control 暴露的稳定性问题

无扰动 control 的 float64 四次全部完成，warm 中位数 0.682 秒。float32 前三次完成；两次完成的 warm 为 0.187、0.191 秒，条件中位数 0.189 秒，**第 3 次 warm（总第 4 次）在初始 H0 的 Cholesky 分解失败**。失败的 0.121 秒只表示中止前用时，完全不计入完成恢复的中位数。

control 的前三个 float32 endpoint 中，两个达到既有 64 位一致性容差，一个目标超额 0.002637，未达到 1e-3。不是所有原生正常停止的结果都满足原数值精度，也不是每次都能完成白化。

失败矩阵及独立 CPU 谱分析见 [control_metric_spectra.json](lbfgs_precision_v1/control_metric_spectra.json)。保存的 32 位 H0 提升到 double 后，仍有约 **-27316、-2839** 的负特征值，CPU double Cholesky 也失败；因此不是只把同一个矩阵交给 64 位分解就能修复。对应 double H0 可正定分解，最大特征值约 4.98e13，最小约 2.27；对角归一化后最小特征值仍只有约 8.56e-10。

这些数值支持强正则化、病态方向与单精度舍入令 H0 丢失正定性的解释。control 的势平滑强度为 1e11，而幅度 prior 为 1e4；某些对角量在 32 位下的相邻浮点间隔已大于这个幅度项。同输入重复恢复表现不同说明分解对计算误差敏感；本实验未逐次保存所有 H0，不把 GPU 累加次序认定为已经隔离证实的唯一原因。

## 判断与复现

当前原样纯 JAX float32 路径可在约 **0.22 秒**完成有扰动恢复，物理场非常接近 64 位，值得保留为提速方向。现有证据支持继续将 **float64 作为稳定默认**。要采用 32 位，还需另行测试更稳定的能量计算、适合单精度的停止逻辑或明确标注的混合精度白化；这些都没有悄悄混进本次结果。

~~~bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python -u workspace/pt_new_algorithm/benchmark_lbfgs_precision.py \
  --output workspace/pt_new_algorithm/lbfgs_precision_recheck
~~~

数值原始记录全部保留，包括失败，不重试择优。测量时脚本完整文本保存在 `lbfgs_precision_v1/runner_at_measurement.py`，SHA256 与四个原始 report 中的 `script_sha256` 一致。运行后修正了一个汇总问题：旧 worker 的 `warm_median_seconds` 曾把中止前时间混进 control 中位数；原始逐次记录未改写，更新的 `summary.json` 按完成情况从原始 rows 重新计算，报告采用其中 `completed_warm_median_seconds`。另缩短了终端汇总输出；拟合计算、精度、参数和验收均未改动。

已完成新脚本的 `py_compile`、Black 检查及 `git diff --check`，并对四个报告的测量源码 SHA、canonical 输入/整数拓扑一致性、32 位实际 dtype/JAXPR 和失败样本不进入完成耗时中位数做断言核验。本次验证以实际大规模拟合和独立数值外验为主，核心库未改动。
