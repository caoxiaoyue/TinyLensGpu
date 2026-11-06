# 标准 Mock：固定100×100 source，把数据降至2312点

2026-11-04，`tinylens_gpu`，RTX 5080，Float64。

**结论：同一曝光的3×3像素合并，将数据从22016点降至2312点后，仍未复现
J0946超过原收敛门槛的停滞。较弱正则化下迭代更多、耗时增加，但所有计划恢复
均通过。** 本次检验参数/数据点数量比；不声称解释了0946的唯一原因。

## 实验条件与对照

直接复用上轮 `source_resolution_v2/frozen_inputs.npz` 的有扰动科学 Mock：
同一seed17、200×200原图、.012 arcsec像素、同一PSF、mask、noise realization、
SIS+shear真值宏模型和Pseudo-Jaffe注入。source保持100×100、bbox=[−.4,.4]²；
potential保持原956个节点、724个core、.075 arcsec间距。总参数固定10956。
source物理尺度匹配prior沿用 `source_resolution_prior.py`；本轮各条件先验逐项相同。

使用原硬边界双线性source插值，不引入零环、高阶插值、参数化subhalo、非负约束、
质量片重标定或新增ridge。细网格射线、原像素积分nsub=2、potential导数、光学PSF
卷积全部保持不变，在观测层施加固定线性算子。

| 观测条件 | 图像布局 | 有效数据点 | 参数/数据点 |
|---|---|---:|---:|
| full：原细图 | 200×200，.012″像素 | 22016 | .498 |
| matched_fine：与粗图同覆盖的细像素 | 200×200，.012″像素 | 20808 | .527 |
| binned：3×3合并 | 66×66，.036″像素 | **2312** | **4.739** |
| 真实J0946，作为数量参照 | 此处不重新拟合 | 2293，11134参数 | 4.856 |

200不能整除3，先中心裁成198，再分为66×66个块；仅保留9个细像素都在原
likelihood mask内的完整块，自然得到2312点，比2293多19点，不另删像素凑数。
matched_fine恰好保留这些块使用的20808个细像素，从而分开检查1208个边缘细像素
被排除的影响和真正像素合并的影响。

## 像素、PSF和噪声的一致处理

定义每行9个权重1/9、且各行支持互不重叠的算子B：

```
d_coarse = B d_fine
m_coarse(q) = B m_fine(q)
C_coarse = B C_fine Bᵀ
```

细像素为独立Gaussian噪声，因此粗观测协方差仍为对角阵：
`σ_coarse = sqrt(sum(σ_i²))/9 = .01/3`。使用平均值保留原亮度单位，而不是将
图像平均后仍填原σ=.01。未对PSF数组做近似重采样；原细网格卷积和像素积分后
再做B，完整包含更大的粗像素响应。potential拓扑/所有细查询坐标hash逐进程核对。

这是同一曝光的真实binning：空间细节和独立观测维数减少，但低频信噪比保留。
例如常值信号的9个细像素与其一个平均值具有相同的总信噪比平方。所以不能把
“数据点少9倍”解释为“所有物理模式的信息都少9倍”。这不是额外加噪声或缩短曝光
的实验，也没有模拟真实0946的全部PSF、噪声结构及source形态。

前向、JAX AD、解析Jacobian/伴随、source初始化和终点检查全部采用同一个B和
同一σ。`noise_logdet`也按新观测定义更新；本轮没有计算或优化evidence。

## 固定强度、初始化和验收

两组profile使用上轮同样的固定值；“evidence_selected”仅指此前原Mock选出的值，
本次没有对粗数据重新选λ。

| profile | λs，49网格参考定义 | λ4 | λ0 |
|---|---:|---:|---:|
| baseline | 6e−4 | 1.6e7 | 1e4 |
| evidence_selected | 7.490065174430917e−5 | 827361.8151935224 | 26790.667792036085 |

100网格的内部/边缘prior采用上轮的物理积分匹配定义；完整解释见
[source分辨率报告](source_resolution_results.md)。像素合并没有改变这些prior。
各自从观测在零修正下的source-only MAP开始，初始化使用相同baseline参考λs。
每次重算source初始化、起点完整AD/GN白化，不使用truth source、上次终点或
相邻条件的终点。固定真值宏模型是Mock原设定，不代表真实数据的宏模型已知。

L-BFGS-B保持 maxiter=3000、maxfun=120000、maxcor=30、maxls=80、ftol=1e−15、
gtol=1e−8。验收保持原生success、有限/SPD、`.5 gᵀH_GN^-1g ≤ .001`；该量是
局部GN二次诊断，不是实际目标差或evidence误差上界。未降低标准或换优化器。
每条件独立进程，一次首次恢复、三次预热重复；三种观测×两组先验，共24次。

## 实测结果

时间和步数为预热后三次中位数；通过数包含该条件的首次恢复，共4次。

| profile / 观测 | 通过 | 步数 | L-BFGS时间 | 恢复流程 | 恢复＋额外GN验收 |
|---|---:|---:|---:|---:|---:|
| baseline / full，22016点 | 4/4 | 187 | .861 s | 4.610 s | 6.991 s |
| baseline / matched_fine，20808点 | 4/4 | 197 | .989 s | 4.676 s | 6.992 s |
| baseline / binned，2312点 | **4/4** | **276** | **1.269 s** | **4.932 s** | **7.288 s** |
| selected / full，22016点 | 4/4 | 521 | 2.372 s | 6.124 s | 8.457 s |
| selected / matched_fine，20808点 | 4/4 | 546 | 2.463 s | 6.154 s | 8.488 s |
| selected / binned，2312点 | **4/4** | **832** | **4.062 s** | **7.786 s** | **10.153 s** |

总计24/24通过，粗数据8/8。所有SciPy退出均为relative-F success，没有达到3000步
上限；历史scaled-gradient门也24/24通过。粗数据的两组GN诊断范围分别
1.67e−6–9.23e−6和1.63e−6–3.08e−6，明显小于.001。
较弱先验的三个预热粗图恢复为650、832、1059步，6.785–8.941秒。

“恢复流程”包含source初始化、初始metric/Cholesky、梯度准备、L-BFGS与图像/场
导出前的计算，不含Mock生成、Python/GPU启动、文件IO和额外完整GN终点资格检查。
各阶段同步计时，BLAS线程固定1。首次恢复包含首次拟合JIT：baseline三条件约
9.60/9.71/10.38秒；selected约10.74/11.18/10.88秒（full/matched/binned）。

粗图没有明显加速：实际还保留细网格的射线/PSF运算，10956×10956白化矩阵也
没有变小。初始化约1.37–1.38秒、初始metric约1.71–1.76秒、分解约.59–.60秒，
各条件相近。为了保持物理前向严格一致，本研究没有实现原生66×66的加速前向。
包括全部恢复、额外检查与IO，正式六个进程总计约270.6秒。

## κ场、真值误差和重复性

| profile / 观测 | 四次κ峰范围 | 残差RMS/noise | 全core κ真值RMSE，约 |
|---|---:|---:|---:|
| baseline / full | .301608–.301621 | 1.00168 | .027064 |
| baseline / matched_fine | .3012965–.3012966 | 1.00214 | .027570 |
| baseline / binned | **.2979045–.2979047** | **1.01714** | **.027736** |
| selected / full | .471322–.471330 | .992007 | .038175 |
| selected / matched_fine | .469726–.469749 | .992028 | .038824 |
| selected / binned | **.461557–.461560** | **.927960** | **.038147** |

所有峰位置仍为(.75,.075)arcsec，距注入(.75,.1)为.025arcsec。
两组粗图source相对L2真值误差约6.624%和6.650%，与细图的6.744%和6.779%处于
同一量级；不能用这一个噪声实现的细微改善宣称低分辨率普遍更好。
truth只在24个终点全部冻结后采样作诊断，没有用于初始化、λ选择或挑解。

离散注入κ峰仍是.631435，粗图峰约恢复其47.2%/73.1%；正则化偏差没有消失。
数值收敛不等于无偏恢复质量或检测显著性。selected粗图残差RMS≈.928也不意味着
噪声模型或超参数已最优，本轮保持λ就是为了隔离数据改变的影响。

相同条件重复的全core最大κ差：baseline粗图1.19e−6，selected粗图7.36e−6；
六组最大值6.59e−5。这些是同一数据的技术重复，不是独立噪声或参数后验样本。
mask裁切虽只轻微改变峰值，完整κ场并非不变：full→matched的core RMS差约
.00303/.00363；matched→binned约.00214/.00530。因此保留matched对照，不能把
full→binned的所有场差异都归于像素合并。

## 对0946原因的约束

本次不仅有约一万个source参数，而且观测数量与参数/数据点比例都接近0946，
仍全部通过同一验收。因此**参数/数据点数量比本身不是本例中产生0946停滞的充分
条件**；较弱正则化下代价上升说明数据约束改变确实会影响优化难度。

这并未证明比例在真实0946中完全无影响。Mock仍是同一个光滑source、正确固定宏
模型、已知Gaussian噪声、uniform second-order prior，并在实际source-only起点
构造白化；0946的source、adaptive first-order prior、模型误差和M3白化参考等
差别没有在此隔离。物理binning还保留了低频信噪比，不能用“点数相等”替代完整
信息结构的匹配。本轮没有继续更改0946模型。

## 验证和复现

针对降采样/先验的8项测试通过，涵盖显式B、方差传播与对角协方差、常亮度守恒、
伴随、PSF后观测组合、完整lens AD/解析导数、正定先验和物理尺度。
全部24次的独立导出检查显示数据/模型与NumPy重新合并的最大差4.45e−16，noise
差为0；最大解析/AD目标误差3.64e−12、梯度相对误差8.48e−9、白化二次型误差
4.32e−14、初始化正规方程相对残差2.24e−15。输入和被测源码hash前后未变。
新full对照与上一轮n100的core κ最大差仅3.19e−5/4.89e−5，物理模型保持一致。

正式目录为 `data_resolution_v2/`。`data_resolution_v1/` 是保存mask元数据时的
序列化中止，所有条件都在拟合开始前失败，未计入24次；修复记录格式后从头完成
v2，没有依据数值结果调整算法或筛选重复。

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python workspace/pt_new_algorithm/run_data_resolution.py --output workspace/pt_new_algorithm/data_resolution_new
JAX_PLATFORMS=cpu python workspace/pt_new_algorithm/analyze_data_resolution.py workspace/pt_new_algorithm/data_resolution_new
```

- [机器可读汇总](data_resolution_v2/analysis.json)、[全部24次CSV](data_resolution_v2/all_fits.csv)
- [运行协议](data_resolution_v2/protocol.json)、[逐次原生状态及原始场](data_resolution_v2/summary.json)
- [入口](run_data_resolution.py)、[观测组合定义](data_resolution_observation.py)、[只读核对/真值诊断](analyze_data_resolution.py)
