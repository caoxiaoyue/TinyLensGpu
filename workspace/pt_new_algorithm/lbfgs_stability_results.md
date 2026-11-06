# L-BFGS最终稳定性检查：10个独立噪声种子

2026-11-03。按用户确认的范围：种子0–9，各做一次完整有扰动恢复，并增加相同噪声的无扰动对照，共20次L-BFGS。每个种子独立进程；没有筛选种子、重试取最好值或沿用上次最优参数。

**结论：正式0–9全部通过。10/10有扰动恢复、10/10匹配control，20次SciPy正常停止、20次旧梯度门通过、20次数值参考一致性与科学sanity通过；没有异常、未知参考或重跑。** 该结果支持在当前模型/初值/先验范围内，后续采用JAX自动微分＋初始曲率白化＋L-BFGS-B，结束本轮算法选型实验。

原始总判定见 [final.json](lbfgs_stability_10seeds/final.json)、[完整summary](lbfgs_stability_10seeds/summary.json) 和 [统计汇总](lbfgs_stability_10seeds/assessment.json)。10个noise hash互异；每组的两case共用同一noise。

## 模型与算法保持什么

沿用原200×200科学Mock：22016有效像素、2401源变量、956势变量、724个导数有效core节点，源49×49、势间距0.075 arcsec、相同PSF/mask及生成/反演nsub=2。固定同一个SIS+shear宏模型及Pseudo-Jaffe注入，只改变seed控制的独立Gaussian噪声。每个seed的有扰动与control逐像素共享同一噪声。

初始源由各自观测在零势修正下做signed source-only MAP得到，未使用source truth；势从零开始，宏模型仍固定真值。该初值与之前部分truth-node计时实验不同，是更接近实际使用的已有初始化方式。源和势沿用原有符号自由度，未添加非负约束、质量片重标定或新物理ridge。

全部正则化保持原值：source=0.0006，potential分别1.6e7/1e11（subhalo/control），所有势节点幅度项lambda0=10000。L-BFGS-B无bounds、JAX自动微分、每次重新计算初始GN白化与Cholesky，使用原参数：

~~~python
maxiter=3000, maxfun=120000, maxcor=30, maxls=80,
gtol=1e-8, ftol=1e-15
~~~

L-BFGS本身是确定性优化器；这里seed实际改变Mock噪声及观测导出的源初值，而不是只重复设置一个优化器不会使用的随机数种子。各种子的实际noise hash和完整输入均保存。

## 预先登记的验收

[protocol.json](lbfgs_stability_10seeds/protocol.json)在正式种子开始前保存了全部阈值、原模型配置、算法参数及脚本/核心源码快照hash。分别报告以下层次：

1. **优化器状态**：SciPy原始success/status/message；不重新定义native success。
2. **数值一致性**：与同seed、同数据/先验/数值初值的独立GN参考相比，目标超额≤1e-3，图像RMS差≤1e-3倍noise，源相对L2差≤0.005，core kappa RMS差≤1e-5+0.01×参考kappa RMS。另要求AD/解析梯度在GN尺度下差≤1e-8、目标身份误差≤1e-6。这些是事先规定的工程误差容差，不是物理定律或置信水平。
3. **物理sanity**：参数、有效图像及724 core导数有限；图像残差RMS/noise在0.9–1.1；注入case的全core正峰定位在一个网格对角线内（sqrt(2)×0.075 arcsec）；control kappa RMS<0.01；恢复sub峰与同seed control最大绝对core kappa的比值≥5。后两项仅为宽泛异常检查，不是噪声假阳性率或检测显著性，尤其两case的potential prior不同。
4. **旧梯度门**：继续记录atol+1e-8×初始梯度的结果，但不再仅因它未通过就称物理恢复失败。

GN参考用SciPy TRF/LSMR、cached导数、显式初始白化、candidate检查、LSMR1e-8。它既可按原梯度门回调停止，也可按ftol/xtol=1e-13正常停止；保留两种原始状态。reference usable表示数值正常结束、公式核对和sanity通过，并不证明全局最优或一定通过旧梯度门。参考失败或明显不一致时保留为未知/需核查，仍执行L-BFGS，不偷偷更换参考结果。

## 结果

| seed | sub峰值 | sub残差RMS/noise | control κ RMS | sub图像对参考Δ/noise | 原生/综合验收 |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 0.306201 | 0.999609 | 0.0011269 | 2.284e-05 | 两case通过 |
| 1 | 0.310024 | 0.995970 | 0.0008432 | 1.118e-05 | 两case通过 |
| 2 | 0.306055 | 1.001197 | 0.0012221 | 1.510e-05 | 两case通过 |
| 3 | 0.306763 | 1.001584 | 0.0009716 | 3.051e-05 | 两case通过 |
| 4 | 0.292982 | 0.999965 | 0.0009398 | 2.873e-05 | 两case通过 |
| 5 | 0.311745 | 1.001091 | 0.0010183 | 1.469e-05 | 两case通过 |
| 6 | 0.305273 | 1.003762 | 0.0015116 | 3.116e-05 | 两case通过 |
| 7 | 0.308876 | 0.999131 | 0.0008854 | 1.921e-05 | 两case通过 |
| 8 | 0.312172 | 0.996725 | 0.0005434 | 1.133e-05 | 两case通过 |
| 9 | 0.307137 | 1.000929 | 0.0007038 | 3.739e-05 | 两case通过 |

全部sub峰位置均为 **(0.75,0.075) arcsec**，距注入(0.75,0.1)为0.025 arcsec；峰范围0.292982–0.312172，平均0.306723。control core kappa RMS范围0.000543427–0.00151160，随独立噪声变化，不能把seed17单点的0.000533当成所有noise的固定幅度。

有扰动残差RMS/noise为0.995970–1.003762，control为0.988889–0.996367。源对真值相对L2误差分别约6.08%–7.36%和4.17%–5.73%，这是含固定离散化/正则化的恢复误差，不是优化器间的数值差。

对同seed GN参考，20次中最大图像差仅3.739e-5倍noise RMS、最大源相对差7.324e-5，最大sub kappa RMS差3.288e-5；最大目标超额只有2.880e-6，均显著小于预登记容差。AD/解析梯度在GN尺度下最大差3.875e-13，目标身份核对也通过。

10个sub的旧缩放梯度范围5.642e-5–2.933e-4，10个control为2.724e-8–7.179e-6，本批确实全部满足旧门，但这不是重新把旧门当作唯一成功定义。

20个参考终点中18个因梯度门回调停止，seed3和seed5的sub参考按native ftol正常停止（status2），未过其旧梯度门；按预登记仍属可用的成本稳定参考，原始flags未改写。不能把“全部参考可用”误称为“全部参考通过旧梯度门”。

本批20次L-BFGS均从原source-only初值重新开始，未从参考解、之前seed解或最优重复点启动。数据、初值、native状态、实际场和源码快照另经独立只读审查核验。


## 解读范围

本测试检验固定系统、固定宏模型与先验下，独立噪声及相应观测源初值变化的稳定性；不是10种不同透镜系统，也没有扫描宏模型错误、势网格、正则化权重或随机远距离初值。

原模型的正则化峰值压低仍需如实报告：离散注入峰约0.631435，而既有正则化解峰约0.30。稳定恢复正则化MAP，不等于恢复了无偏扰动质量或完成检测显著性校准。

这是新数据/新进程的稳定性测试，包含首次编译；不使用这些首次运行耗时替代之前的warm性能比较。

## 复现与原始记录

~~~bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python -u workspace/pt_new_algorithm/run_lbfgs_stability.py \
  --output workspace/pt_new_algorithm/lbfgs_stability_recheck
~~~

入口：[run_lbfgs_stability.py](run_lbfgs_stability.py)。[独立设计/源码/结果审查](lbfgs_stability_review.md)。每seed保存config、noise hash、初值、native状态、history、endpoint、完整图像/源/势/真值场以及同seed GN参考。

seed17仅用于新脚本预检，不纳入正式0–9的通过率。第一次预检在保存NumPy布尔量时出错，求解endpoint和原错误记录保留于lbfgs_stability_pilot；修复JSON标量转换后在新目录lbfgs_stability_pilot_v2的两例通过。没有因此改变任何物理/数值阈值。旧的L-BFGS梯度门未通过记录仍保留在历史计时目录。
