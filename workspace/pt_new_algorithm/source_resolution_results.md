# 标准 Mock：source 49×49 → 100×100 的受控实验

2026-11-04，`tinylens_gpu`，RTX 5080，Float64。

**结论：在物理正则化尺度匹配后，增加 source 参数没有在这个 Mock 上复现
J0946 的验收失败。主要代价是矩阵计算更贵，而非需要更多 L-BFGS 迭代。**
两组固定强度、两种分辨率，每条件一次首次恢复加三次预热重复，共16次，全部通过
原生 SciPy success 和同一个 `.5 gᵀH_GN^-1g ≤ .001` 门槛。该门槛是局部二次
模型诊断，不是全局最优证明或实际目标差上界。

## 保持固定的条件

使用既有200×200有扰动科学 Mock，seed17、dpix=.012、nsub=2；同一个
SIS+shear 真值宏模型、Pseudo-Jaffe 注入、PSF27×27、mask、noise。数据只生成
一次并冻结为 `source_resolution_v2/frozen_inputs.npz`，每个进程直接读取，逐数组
校验；image_data/noise 和此前 DIRECT 实验的保存数据逐元素完全一致。

| 量 | 49×49 | 100×100 |
|---|---:|---:|
| source 自由参数 | 2401 | 10000 |
| potential 自由参数 | 956 | 956 |
| 联合参数 | 3357 | 10956 |
| 数据像素 | 22016 | 22016 |
| potential spacing | .075 arcsec | .075 arcsec |
| source bbox | [−.4,.4]² arcsec | [−.4,.4]² arcsec |
| source spacing | .0166667 arcsec | .00808081 arcsec |

每个 worker 校验 potential 节点坐标、拓扑、724个 core、子像素查询坐标和宏模型
回溯坐标的 hash 全部一致。原来的硬截零双线性 source 插值保持不变，未加入零环、
高阶插值、非负约束、参数化 subhalo、gauge 投影或质量片 rescaling。

两组强度事先固定，无新超参数搜索：

| profile | λs，49网格参考定义 | λ4 | λ0 |
|---|---:|---:|---:|
| baseline（原稳定性测试） | 6.0e−4 | 1.6e7 | 1.0e4 |
| evidence_selected（此前 DIRECT 结果） | 7.490065174430917e−5 | 827361.8151935224 | 26790.667792036085 |

source 初值均只使用观测：在零 potential correction 下求 signed source MAP。
两组 profile 的初始化都沿用原 DIRECT 的 baseline 参考 λs=6e−4，并使用相应
分辨率的匹配先验；不使用 source truth、不从旧最优点或上一次重复解出发。
每次重新构造完整初始 AD/GN metric 和 Cholesky 白化，不缓存数值矩阵。
L-BFGS-B 设置保持 maxiter=3000、maxfun=120000、maxcor=30、maxls=80、
ftol=1e−15、gtol=1e−8；没有 LM/GN 更新、jitter、失败后重试或调松门槛。

## 正则化物理尺度如何匹配

现有 uniform second-order source penalty 含 `h^-4`，内部为二阶差分，靠近最后
边缘为一阶差分，最后边缘为亮度 fallback。仅修改统一 λ 或仅保留 h^-4 都不能
让这三类项同时对应同样的物理系数。

设 `h*=.8/48`，`r=h/h*=48/(n−1)`。本实验在 `source_resolution_prior.py`
中定义 workspace prior：

```
λs*/h^4 × [r² Σ内部(D²s)² + r³ Σ边缘(Ds)² + r⁵ Σ边缘s²]
```

内部曲率对应二维面积积分，边缘梯度/亮度对应一维线积分。三个连续系数分别为
`λs*/h*²`、`λs*/h*³`、`λs*/h*⁵`，两种网格相同。n=49时 r=1，完整恢复旧
prior，包括原来只在最后行/列使用 fallback 的非对称边界 convention。
λs* 是固定的49网格参考强度；100网格等效内部系数为 `λs*×(48/99)²`，
baseline 约1.41046832e−4，另需上述边缘项相对缩放。λ4和λ0完全不变。

这是同一连续惩罚的两种离散近似，不要求任意离散场的 penalty 数值逐项相同。
常数、线性、二次多项式的对应分量核对在1.5%以内（端点求积误差）。同一个物理
高斯测试场的总 penalty 从1.36999054e6变为1.37320415e6，仅差约0.235%；其中
曲率项差0.62%、边缘亮度差约1e−6相对量。边缘梯度项差约11%，来自有限差分
取样位置/截断误差；该项仅占总量约2%，并未改变其连续系数。若完全不匹配，
同一场100网格的 penalty 为1.66212571e7，约是49网格的12.1倍。

## 收敛与耗时

下表只对预热后的3次取中位数；首次恢复另列。完整恢复包含 source 初始化、
初始 metric、Cholesky、梯度准备、L-BFGS 和恢复图像/场；不含额外研究验收、
文件导出、Mock生成、Python/GPU启动。每个阶段都同步计时，BLAS线程数固定为1。

| profile / 网格 | 预热严格通过 | L-BFGS步数中位数 | L-BFGS秒数 | 完整恢复秒数 | 首次恢复秒数 |
|---|---:|---:|---:|---:|---:|
| baseline / 49 | 3/3 | 239 | .389 | 1.017 | 5.912 |
| baseline / 100 | 3/3 | 216 | .993 | 4.667 | 9.684 |
| evidence_selected / 49 | 3/3 | 682 | 1.081 | 1.736 | 6.519 |
| evidence_selected / 100 | 3/3 | 451 | 2.129 | 5.827 | 10.545 |

四次首次恢复也全部通过，因此总计16/16，100网格为8/8。全部 SciPy 正常按
relative-F条件结束，没有到达迭代预算。100网格的步数没有增加；总恢复耗时分别
增加4.59倍和3.36倍，L-BFGS阶段增加2.55倍和1.97倍。

| 阶段，预热中位数 | baseline 49 | baseline 100 | selected 49 | selected 100 |
|---|---:|---:|---:|---:|
| source 初始化 | .170 s | 1.371 s | .167 s | 1.371 s |
| 初始 AD/GN metric | .437 s | 1.738 s | .448 s | 1.709 s |
| Cholesky 白化分解 | .0257 s | .593 s | .0259 s | .595 s |
| L-BFGS | .389 s | .993 s | 1.081 s | 2.129 s |
| 额外终点完整GN检查 | .474 s | 2.328 s | .482 s | 2.326 s |

各阶段中位数相加不必等于总时长的中位数。若把额外完整 GN 验收也纳入每次
服务时长，还应加最后一行；这里未计算 Bayesian evidence，也未把它混入计时。
本轮所有进程、16次恢复及检查和导出共约130秒，原始日志均保留。

| profile / 网格 | 四次GN stationarity范围，须≤.001 | κ峰值范围 | 残差 RMS/noise |
|---|---:|---:|---:|
| baseline / 49 | 1.32e−5–6.83e−5 | .3035769–.3035777 | 1.001837 |
| baseline / 100 | 3.06e−8–1.03e−7 | .3016121–.3016195 | 1.001675 |
| evidence_selected / 49 | 5.17e−6–1.06e−5 | .4766803–.4766993 | .992125 |
| evidence_selected / 100 | 3.93e−7–2.46e−6 | .4713236–.4713389 | .992007 |

所有峰位置仍为(.75,.075)arcsec。相同网格/强度的重复间，全core κ最大差不超过
4.47e−5，图像RMS差不超过1.16e−5倍noise。不同网格的图像差约.00333/.00268倍
noise，core κ RMS差约.00128/.00146；改变离散近似后不要求逐节点相同。
这些差异不能当作后验不确定度，也不表示100网格的物理恢复必然更好。

同时保留历史的 scaled-gradient 门：baseline49仅2/4通过该旧门，其余三条件
各4/4。16/16指统一的原生状态＋完整GN诊断验收；未将旧门改名、删除或声称
每个终点梯度精确归零。参数维度变化后，只比较原生 success 也不够。

## 对0946原因的意义与边界

在同一数据/物理先验下，把总参数从3357增至10956、source增至10000，并未让
本例出现0946那种超过统一门槛的停滞。**参数总数增加本身不足以解释0946。**
它确实显著增加初始化、矩阵构造、分解和每步运算成本。

只读方向探测还发现：Mock的四个首次终点同样可能距内部格线仅1e−11–1e−8格，
跨格也可能让负梯度预言与实际小步变化不符。但本例GN诊断最大仅6.83e−5，已经
小于.001，而此前0946固定λ实验为.0065–.020。这要求收窄此前解释：**存在内部
插值折点不是0946失败的充分证据；折点影响的大小及问题的约束结构更关键。**
四组Mock方向探测均无原bbox跨越；原始数值保留，极小步长接近浮点噪声时不把
有限差分符号当成进一步的精确证明。

此实验仅一个物理系统、一个固定噪声 realization、两组先验的技术重复。物理先验
匹配会抑制新加入的高频自由度，所以不能推广为任意高维问题都容易。
0946还同时有更少数据（2293像素）、不同source prior（adaptive first-order）、
模型失配及不同白化参考等差别。本轮沿用Mock在实际source-only起点构造metric的
流程；0946上一轮四条件共享保存M3参考metric，未在这里隔离该差别。
本例结果没有证明0946的唯一主因，也没有排除参数相对数据约束的影响。

## 验证与复现

新prior的4项针对性测试通过，覆盖49原prior等价、dense/matvec/AD一致、对称正定、
跨分辨率物理积分核对。16次真实恢复最大解析/AD目标误差3.64e−12、梯度相对误差
8.43e−9，白化二次型误差5.84e−14，source初始化正规方程相对残差2.20e−15。
协议中的源码及原配置/旧selected结果hash前后核对，未改生产库默认行为。

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python workspace/pt_new_algorithm/run_source_resolution.py --output workspace/pt_new_algorithm/source_resolution_new
python workspace/pt_new_algorithm/analyze_source_resolution.py workspace/pt_new_algorithm/source_resolution_new
```

正式结果为 `source_resolution_v2/`，包含协议、源码快照、冻结输入、逐次初值/终点/
history/原生状态、环境和完整计时。`source_resolution_v1/` 是准备输入时误用
PotentialGrid属性名导致的中止尝试，没有执行拟合；修正为 `n_unknown` 后从头完成
v2，不混用输出。所有计划条件均保留，没有挑选最好的一次。

- [机器可读汇总](source_resolution_v2/analysis.json)
- [全部16次与分段计时CSV](source_resolution_v2/all_fits.csv)
- [运行协议](source_resolution_v2/protocol.json)
- [运行入口](run_source_resolution.py)、[prior定义](source_resolution_prior.py)、[只读汇总程序](analyze_source_resolution.py)
