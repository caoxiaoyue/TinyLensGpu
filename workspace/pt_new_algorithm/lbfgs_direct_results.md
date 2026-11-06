# DIRECT 三超参数搜索，内层 JAX / L-BFGS 势修正

2026-11-04，标准 RTX 5080 科学 Mock。

入口：[run_lbfgs_direct.py](run_lbfgs_direct.py)。正式记录保存在
[lbfgs_direct_full_v1](lbfgs_direct_full_v1/protocol.json)；
[独立数学与结果审查](lbfgs_direct_evidence_review.md)。

**这套两层结构已经跑通。** 两个 case 各 305 次 DIRECT 调用，有扰动搜索
8.95 分钟、无扰动 3.35 分钟。三强度选择显著提高各自数据下的 local-GN
evidence；有扰动峰值更接近真值、整体 κ 误差却变大，无扰动的伪修正更小。
两次 DIRECT 均以预算耗尽结束，保留的是预算内最佳可用候选。

## 两层计算与 evidence 定义

外层以 `log10(lambda_source), log10(lambda_potential), log10(lambda_amplitude)`
为坐标调用 `scipy.optimize.direct`。内层固定这一组三个强度，使用 JAX 自动微分、
初始完整曲率白化和 SciPy L-BFGS-B，从同一初态独立恢复源 s 与势修正 p。
本次外层没有追加超参数的 L-BFGS 局部精修；L-BFGS 只用于内层场恢复。

沿用已有的完整先验与物理变量积分测度：

\[
Q=\operatorname{blockdiag}\left(\lambda_s R_s,
\lambda_4D_4^TD_4+\lambda_0I\right),\qquad
F(q)=\tfrac12\left(\|(d-m(q))/\sigma\|^2+q^TQq\right).
\]

每个候选的 evidence 在恢复终点重新计算：

\[
H_{\rm GN}=J(\hat q)^TWJ(\hat q)+Q,
\]
\[
\log E_{\rm GN}=-F(\hat q)
+\tfrac12\left(\log\det Q-\log\det H_{\rm GN}
-\log\det C-N_d\log(2\pi)\right).
\]

这与库中 iterative 的 `local_gauss_newton` 定义一致，是**非线性模型的局部
GN 高斯近似**，不是精确非线性边际化，也不是使用完整 Hessian 的 Laplace。
完整联合终点曲率保留源—势耦合，使用 full Cholesky determinant；没有采用
operator block determinant、L-BFGS 的 `hess_inv` 或初始曲率 determinant。

lambda0 同时进入每个势节点的 penalty、白化和先验归一化。先验 determinant
采用精确的块对角分解：`Ns*log(lambda_s)+logdet(Rs)` 加完整
`logdet(lambda4*R4+lambda0*I)`；后者不能简化为 `Np*log(lambda4)`。
当前 Rs 含既有边界 fallback、为 proper prior；没有添加新 ridge 或硬 gauge。
source/potential 都保留原有符号自由度，未做非负裁剪或质量片重标定。

## 模型、初始化和冻结协议

标准 200×200、seed17、dpix=0.012、生成/反演 nsub=2，源49×49，势间距
0.075 arcsec；22016 数据、2401 源变量、956 势变量、724 有效 core 节点。
保留原 SIS+shear 宏模型和匹配 Pseudo-Jaffe/control Mock；宏模型固定真值。
全部数值计算使用 float64，RTX 5080，JAX matmul precision=highest，BLAS=1。

默认初始源只由观测在零势修正下做一次 signed source-only MAP 得到，使用原
baseline 的 lambda_s=0.0006。这个初始源之后固定，对所有超参数候选相同；
势始终从零开始。source truth 不参与初始化或选择，只在冻结 baseline/selected
候选之后计算诊断。与早期约1秒的 truth-node 初始化基准应区分。脚本保留
`--initialization truth`，本轮正式运行使用默认 source_only。

外层 log10 区间沿用旧三参数实验：source [-8,-1]、D4 [1,11]、amplitude [0,8]。
每 case 的 DIRECT 参数为 maxfun=300、maxiter=1000、eps=1e-5、len_tol=1e-3、
vol_tol=1e-16、locally_biased=True。maxfun 为近似预算，实际 nfev 另行记录；
区间和搜索预算不是全局最优证书。

内层保持原参数：maxiter=3000、maxfun=120000、maxcor=30、maxls=80、
gtol=1e-8、ftol=1e-15。接纳要求 native success、有限值、先验和初始/终点
曲率可正定分解，以及 `0.5*g.T@H_GN^-1@g <= 1e-3`。最后一项只是 GN
局部二次模型预测可降幅，不是真实剩余目标 gap 的上界，也不能保证 evidence
误差≤1e-3。旧 scaled-gradient 门单独报告，不是唯一科学判据。
不通过的候选保留原因、不给 evidence、向 DIRECT 返回正无穷代价；没有
LM/GN 求解回退、改变阈值、真值筛选或从上一个候选 warm start。

仅复用固定几何/数据、先验基矩阵与 logdetRs、同一 q0 的初始数据 Gram。
每个候选仍重新构造 Q、初始 H0 与分解，并执行完整 L-BFGS；终点 J/H
必须刷新。白化的对角 equilibration 是数值坐标缩放，不是新物理 prior。
编译函数把三个强度作为动态数组传入，避免每候选重新 JIT。

## 公式和重复验证

正式搜索之前，两例均独立检查同状态下 AD evidence 与原解析 dense backend。
验证时 fresh constructor 显式传入该候选 lambda0，避免浅拷贝继承旧绑定 JIT。
验证不运行 LM，也不替换 L-BFGS 终点。最终所选候选再做同样复核。

baseline 另做一次独立完整恢复；所选点再独立恢复两次。所有重复均从相同
初态开始，不替换搜索所选结果。预先登记的重复 evidence 差容差为 0.05，
所选点是否满足另行保留。它只检验数值可重复性，不证明全局唯一性。

早期预检 `lbfgs_direct_preflight_v1` 在开始数值计算前遇到 Path 的 JSON
序列化问题；修复后 preflight_v2 两例通过。其 baseline 重复 evidence 差
约 8.61e-5 / 1.22e-8，同状态解析公式差约 1.16e-10 / 1.00e-7。
正式运行保留自己的独立 baseline、重复、源码及依赖 SHA 和原始候选日志。

## 结果与复现

有扰动部分已完成。DIRECT 实际调用 305 次，29 轮，以 maxfun 预算耗尽结束；
不是外层原生收敛，也不证明全局最大。加 baseline 共记录 306 个不同候选，
267 个可用，39 个拒绝（17 次内层 native failure、22 次 GN 预测可降幅超限）。
实际 DIRECT loop 用时 **536.77 秒（8.95 分钟）**，包含候选求解与其日志/终点保存，
不含之前的一次性准备、baseline 公式核验和之后的所选解复核。

| 有扰动数据 | 原固定权重 | DIRECT 保留的最佳候选 |
| --- | ---: | ---: |
| lambda_s | 6.0e-4 | **7.49007e-5** |
| lambda4 | 1.6e7 | **8.27362e5** |
| lambda0 | 1.0e4 | **2.67907e4** |
| local-GN log evidence | 69037.256792 | **69538.883617** |
| κ 正峰 | 0.303577 | **0.476701** |
| 残差 RMS/noise | 1.00184 | 0.992125 |
| 源相对 L2 误差 | 7.02874% | 6.98025% |
| 势 RMSE，对去均值真值的诊断比较 | 0.00349800 | 0.00344418 |
| 偏折向量 RMSE | 0.00728014 | 0.00805339 |
| 全 core κ RMSE | 0.0267763 | **0.0382699** |
| local core κ RMSE | 0.0582775 | **0.0619801** |

三个所选强度均远离边界，位置仍在 (0.75,0.075) arcsec。离散真值峰约
0.631435，峰值恢复由约 48.1% 提升到约 75.5%；但全场与局部 κ RMSE 均增大。
势本身的上述 RMSE 略降，偏折误差增加；不能将 κ 导数场的误差笼统等同于
势本身的误差。真值去均值只是既有 gauge 诊断，实际拟合场和 evidence 均未改动。
**evidence 提升 501.626825 与峰值更接近真值，并不等于所有恢复误差都改善。**
本次选择目标始终是 evidence，没有用真值 RMSE 再筛选或替换候选。

所选点的独立解析公式差约 5.82e-11；两次独立 L-BFGS 重跑的 evidence 差分别
+0.002445、-0.001789，均满足预登记的 0.05 重复容差。baseline 重跑的 evidence
差为 -0.0001696；该重跑旧 gradient 门未通过，但 native 与预登记 GN 工程门
通过，不据此重新定义失败或修改外层目标。

无扰动对照也完成 305 次 DIRECT 调用，36 轮、同样以预算耗尽结束。加 baseline
共 306 候选，291 个可用，15 个拒绝（4 次 native failure、11 次 GN 预测可降幅超限）。
DIRECT loop 用时 **201.20 秒（3.35 分钟）**。

| 无扰动数据 | 原固定权重 | DIRECT 保留的最佳候选 |
| --- | ---: | ---: |
| lambda_s | 6.0e-4 | **8.55261e-5** |
| lambda4 | 1.0e11 | **8.67504e10** |
| lambda0 | 1.0e4 | **8.92519e7** |
| local-GN log evidence | 69487.837765 | **69808.500143** |
| 全 core κ RMS | 0.000532938 | **0.0000763851** |
| 残差 RMS/noise | 0.995828 | 0.993612 |
| 源相对 L2 误差 | 5.17100% | 5.51242% |

同一 control 数据下 evidence 提升 **320.662379**，伪 κ RMS 减小约 7 倍，
但源真值误差略增。lambda4/lambda0 距各自上界仅 0.06173/0.04938 dex，
按预登记的归一化边界距离规则均标为近边界；不能称已识别出有限内点最优强度。
它们可能仍偏好更强的抑制，本次没有事后扩边或加入另一个超参数优化器。

control 所选点两次重跑 evidence 差仅 ±1.46e-11，解析公式差约 2.04e-10。
两 case 所选候选及重复均通过预登记数值门；不是声称所有搜索候选都成功。

### 计算量与时间分解

全部 612 行对应 612 个不同三维候选（每例 305 个 DIRECT 候选加一个 baseline），
均保留终点文件（包含失败诊断）；558 个数值接纳，54 个拒绝。三次额外恢复/每 case
（一次 baseline repeat、两次 selected repeat）另记，不算进 DIRECT 调用预算。
外层数值接纳与 DIRECT 的原生预算退出状态分开保存。

| 候选计时与内层工作量 | 有扰动 | 无扰动 |
| --- | ---: | ---: |
| 全候选耗时中位数 | 1.402 s | 0.508 s |
| 可用候选耗时中位数 | **1.374 s** | **0.507 s** |
| 可用候选 L-BFGS 时间中位数 | 0.877 s | 0.0116 s |
| 可用候选终点曲率时间中位数 | 0.465 s | 0.464 s |
| 可用候选 Q/H0/分解时间中位数 | 0.0222 s | 0.0222 s |
| SciPy 记录的内层 F/梯度调用数，含 baseline 与拒绝候选 | 285689 | 39323 |

内层调用数取 SciPy `nfev`，不包含额外 JIT 准备和终点诊断调用。
各阶段中位数之和不一定等于总中位数。源/势更弱正则化的候选通常比最初固定
强度的基准需要更多内层迭代；无扰动最强抑制区域则很快，终点曲率成为主要成本。
初始数据 Gram 的复用节省了重复准备，但 evidence 必须重新计算终点曲率，
因此不能把本表直接当作“原约1秒 L-BFGS 恢复”的同范围速度比较。

从 worker 库导入结束计起，完整主要流程（生成/初始化、准备、公式审查、搜索、
最终导出与重复）分别为 558.10 / 219.06 秒，合计约 12.95 分钟；不包含解释器和
import 启动。失败候选的计算时间保留在全量统计中，未当成合格恢复的加速。

### 冻结结果图

![势修正比较](lbfgs_direct_full_v1/figures/correction_comparison.png)

同一行共用色阶，上行为注入 case，下行为无注入 control；两行色阶不同并各自
明确标注。左固定强度，右 DIRECT 选择。黑叉仅表示上行的注入位置。绘制全部
724 个导数有效 core，956 个势变量全部参与拟合；没有重心对齐、去均值、裁剪
或插值平滑。这些是势修正的 κ 导数图，原始势和源也保存在 NPZ 中。

![搜索过程](lbfgs_direct_full_v1/figures/search_history.png)

每个 case 的纵轴相对它自己的 baseline，使用 symlog（线性阈值10）保留全部
有限候选分数。底部短线是拒绝候选的索引，不赋予虚构 evidence；线为累计最佳。
完整候选表另存为两个 CSV。图中 plateau 只是本预算的表现，不是全局收敛证书。

图像另有 editable PDF/SVG 与 600 dpi TIFF。渲染后的两图面板对齐、PDF 字号
和碰撞检查均通过，最小实际字号分别 8.0 / 5.6 pt，碰撞均为 0 fail/0 warn。
源码审查的 width 警告误将 `180/25.4` 的180视作英寸，实际 PDF 宽度核对为180 mm；
其余源码检查通过。已逐面板检查图、共享色阶和拒绝标记语义。

### 验证与复现

新增用户要求的 [六联图绘图入口](plot_lbfgs_direct_six_panel.py)，直接读取保存的
evidence-selected 结果，不重新拟合：

![图像和扰动六联图](lbfgs_direct_full_v1/figures/subhalo_selected_six_panel.png)

上排为 data、nonlinear model、`(data-model)/noise`；下排为注入 δκ、恢复 δκ、
`恢复−注入`。data/model 共用强度色阶，注入/恢复共用 κ 色阶；归一化残差与
κ 差值各有独立对称色阶，全部显示值都在色阶内。黑叉为扰动注入位置。
κ 指扰动修正，不包含宏模型的 κ。注入 κ 也由同一 0.075 arcsec 势网格的
中心差分获得；解析 κ 另保留在 source NPZ 中，没有把不同离散化的导数混减。

图像只显示实际拟合的 22016 像素；其余17984像素留白。κ 显示724个有效 core，
232个 support-only 节点没有有效导数、留白。读取旧 Mock 真值前核对了 data、
noise、mask、整数拓扑与所有节点坐标完全一致；独立 NumPy 差分也复核了注入/
恢复 κ。归一化残差 RMS=0.992125，差值 κ RMS=0.0382699，与正式报告一致。

六联图有 [PDF](lbfgs_direct_full_v1/figures/subhalo_selected_six_panel.pdf)、
SVG、600dpi TIFF 和 [精确绘图数组](lbfgs_direct_full_v1/figures/subhalo_selected_six_panel_source.npz)。
源码审查21项通过；实际183×153mm，PDF最小字号8pt，面板对齐与碰撞检查通过。
manifest、source/font/collision/alignment 审查及综合 qa JSON 与图片同目录保存。

~~~bash
python workspace/pt_new_algorithm/plot_lbfgs_direct_six_panel.py \
  workspace/pt_new_algorithm/lbfgs_direct_full_v1
~~~

两个所选点均与 fresh 原解析 backend 做同状态完整归一化 evidence/梯度验证。
另以独立 NumPy stencils 重新计算 baseline/selected 的 prior penalty 与 F，
误差在 1e-12 量级。所有 612 行公式、数值接纳、初态与源码/依赖 hash 和
所选 max 记录均独立只读核验。新脚本与绘图入口通过 py_compile、Black 与
`git diff --check`；核心库没有修改。正式搜索中没有按结果重跑或放宽门槛。

~~~bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python -u workspace/pt_new_algorithm/run_lbfgs_direct.py \
  --output workspace/pt_new_algorithm/lbfgs_direct_recheck --maxfun 300
python workspace/pt_new_algorithm/plot_lbfgs_direct.py \
  workspace/pt_new_algorithm/lbfgs_direct_recheck
~~~

每个 case 保留 evaluations.jsonl、所有可用终点、best/summary、baseline 和
selected 原始场、恢复诊断、same-state 公式审查和独立重复。一次性初始化与
engine 准备（包括准备文件写盘）单列；候选耗时包含新 Q/H0/分解、优化、
终点曲率与 evidence，排除候选 NPZ/日志写盘和最终图像导出。

结论应限定于固定宏模型/网格/数据/搜索区间的经验贝叶斯点选择。evidence
提高不保证所有真值误差指标都降低；无扰动与有扰动使用不同数据，二者最大
evidence 的差不能用作扰动检测 Bayes factor，也未估计超参数后验区间。
