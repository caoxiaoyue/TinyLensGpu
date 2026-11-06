# L-BFGS混合精度：边界、梯度与结果核查

2026-11-03。只读主源、runner和保存数据；不运行GPU、拟合或竞争benchmark。本轮按用户授权：H0构建及Cholesky用64位，其余forward/物理AD梯度用32位，物理prior保持原样。

## 两条需要分开报告的精度路径

先在canonical64 problem、相同初始状态和prior上真正计算

\[
H_{64}=J_{64}^TW_{64}J_{64}+Q_{64},\qquad H_{64}=L_{64}L_{64}^T.
\]

把32位算出的H0提升为64再分解，不能代替上述构建。上一轮失败H32转double后仍有负特征值，已证明仅提高分解运算精度不足：[原精度审查](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_precision_review.md)。原[AD metric与解码](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/benchmark_scientific.py:47) 给出完整物理H0与\(q=q_0+L^{-T}z\)，不是只用prior或对角H。

| 路径 | 解码及forward | 白化梯度 |
| --- | --- | --- |
| mixed32：64分解后cast因子 | \(L_{32}=\mathrm{cast}_{32}(L_{64})\)，\(q_{32}=q_{0,32}+\mathrm{solve}(L_{32}^T,z_{32})\)，F32及gq32 | \(g_{z,32}=\mathrm{solve}(L_{32},g_{q,32})\) |
| mixed64-solve：保留64白化 | \(q_{64}=q_{0,64}+\mathrm{solve}(L_{64}^T,z_{64})\)，cast q32后计算F32/gq32 | \(g_{z,64}=\mathrm{solve}(L_{64},\mathrm{cast}_{64}(g_{q,32}))\) |

两者都满足“forward与物理gradient32”，但后者每次仍有64位解码/转置求解，不能合并计成全部其余32。JAX对float cast的JVP转换切线dtype、转置规则把cotangent转换回输入dtype：[已安装规则](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/jax/_src/lax/lax.py:4860)。因此mixed64白化gradient64本身不意味着其物理图像gradient是64。

原始SciPy L-BFGS-B的CPU x/f/g/work为float64，是明确的API边界，不是JAX计算精度：[SciPy源码](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/scipy/optimize/_lbfgsb_py.py:442)。应逐项记录H、L64、实际传入解码因子、z、解码q、forward、prior、物理gradient、白化gradient及FFT的dtype；只看输出图像或JAX总x64开关不够。[JAX dtype规则](https://docs.jax.dev/en/latest/101/default_dtypes.html)。

## 改善什么、仍不保证什么

从64位正定H0分解再cast L，避免先形成已损失正定性的H32。只要L32对角保持非零，它仍定义一个固定可逆坐标变换，物理prior不因此改变。但32位三角求解和因子舍入仍会受条件数影响，不等同于64位白化精度；保留64求解的控制变体可区分这部分误差。

F32及原q·Qq/gradient的舍入继续存在。F约11735时32位ULP约9.77e-4，原ftol=1e-15不会令它达到64目标分辨率；提高准备阶段精度不能自动修复目标量化、prior强弱项相消或提前native停止。[原objective](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/run_experiment.py:41)、[原prior](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/utils/inversion/potential_regularization.py:66)、[SciPy停止定义](https://docs.scipy.org/doc/scipy-1.17.0/reference/optimize.minimize-lbfgsb.html)。

两条路径都需要用**独立canonical64 renderer/data/prior**重新评估物理终点，并分别保存native停止、legacy gamma和既有图像/源/κ/F容差；不能只把F32标量提升到64，也不因mixed表现放宽原F excess≤1e-3门。修改能量计算、prior强度、添加jitter或事后64精修，均是另一个实验。

## 对照与计时要求

主对照保持标准200²、3357变量、canonical数据/PSF/mask/整数拓扑、truth-node初始源和p0=0，以匹配此前约1秒基线；与观测source-only初始化的十seed实验分开解释。预计算canonical几何后cast32的流程应明确，32与64不执行逐位相同的浮点函数。

每次fit均计入新H64构建、Cholesky、因子cast、L-BFGS、终点诊断及raw products，不复用上次分解或最优解。冷启动与复用kernel后的时间分开；canonical64外验可在计时外，但不能参与停止。报告分别列出完成样本与达到既定质量门的样本；分解失败、预算失败或质量未过的耗时不得混成同精度加速。[JAX同步计时规范](https://docs.jax.dev/en/latest/benchmarking.html)。

本批采用seed17同输入重复，未扩展不同噪声seed；重复次数检验该输入的数值重复性，不增加独立噪声样本数。

## 实现预检与修订后的精度边界

已只读审查 [benchmark_lbfgs_mixed_precision.py](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/benchmark_lbfgs_mixed_precision.py) 及归档快照。v1中64 H0/分解均成功，但mixed32在gradient setup遇到JAX索引lowering的int32/int64冲突，未进入优化；[v1原始记录](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_mixed_precision_v1/seed17_mixed32_subhalo/result.json) 保留。它属于实现预检，不能计作mixed方法收敛失败或与正式重复样本混合。

v2将整个mixed32 composite的AD tracing/lowering置于False精度作用域，物理gradient、当前对角和products也分别在正确作用域执行。mixed64则拆成decode64→q32、独立physical value-and-grad32(False)、`solve(L64, cast64(gq32))`(True)，避免跨精度作用域的复合反向索引lowering。这个固定线性转置链式计算正确；其额外GPU dispatch及64白化成本实际计入，并单独标为控制变体。未更改物理forward/prior、L-BFGS选项或验收阈值。

实际物理JAXPR在两个mixed路径均无f64/c128；H0及原始Cholesky均64，mixed32实际factor/白化gradient为32，mixed64对应64，物理objective/gradient/解码结果均32、FFT complex64。终点对角以32 kernel计算，旧PreparedAD用host64缓冲收集后明确cast32，这只是host存储边界，不能声称全部buffer均为32。

## 正式v2：66个终点的独立CPU核查

[冻结v2快照](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_mixed_precision_v2/runner_at_measurement.py)、[全部summary](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_mixed_precision_v2/summary.json) 与逐worker记录已核验。三个variant×两个case，每组cold+10 warm，共66次均完成、分解成功、native正常停止及physical sanity通过。全部runner/dependency hash、配置、canonical输入与topology hash相符，且与上一轮标准64输入一致；未发现隐藏jitter或复用数值H0。每组11个metric hash互异，证实装配结果有位级差异，但64分解在本批全成功，未隔离具体浮点累加机制。

已从每个raw endpoint与canonical_fields独立重算源/图像/core κ差、κ RMS/峰值、数值阈值及accepted判定；66份dtype、snapshot和summary计数/中位数一致。mixed物理终点均为32数组，再以原64模型解释；canonical外验未反馈停止。

| 每case 11次，含cold | Full64 | Mixed32：cast因子32 | Mixed64：保留solve64 |
| --- | ---: | ---: | ---: |
| subhalo完成/native/分解 | 11/11 | 11/11 | 11/11 |
| control完成/native/分解 | 11/11 | 11/11 | 11/11 |
| subhalo达原canonical门 | 11/11 | 0/11 | 0/11 |
| control达原canonical门 | 11/11 | 11/11 | 11/11 |
| subhalo warm完成中位数 | 0.969980 s | 0.590222 s | 0.608975 s |
| control warm完成中位数 | 0.668875 s | 0.535726 s | 0.553054 s |

主Mixed32观察耗时比约1.64倍/subhalo、1.25倍/control；subhalo比值不是相同F精度的成功加速。H64构建约0.44秒、分解/cast约0.037秒，每次均计入，已占Mixed32大部分成本。此前纯32约0.22秒的时间不能与本轮假设准备阶段免费或缓存的时间混算。各阶段中位数也不应机械加成总中位数。

| 相对canonical64，含cold与warm | Mixed32 subhalo | Mixed64 subhalo |
| --- | ---: | ---: |
| F超额范围 | 0.002934–0.009096 | 0.002893–0.009755 |
| 最大图像差RMS/noise | 0.000585 | 0.000611 |
| 最大源相对L2差 | 0.0680% | 0.0760% |
| 最大core κ差RMS | 0.000306 | 0.000324 |
| 峰值范围 | 0.303141–0.303275 | 0.303147–0.303271 |

两个mixed路径的subhalo图像、源、κ一致性门和定位sanity均11/11通过，但每个终点F超额均超过原1e-3门，故accepted=0/11；未事后抬高门。保留64位白化求解也未消除这个F精度损失，说明本次误差不能只归于factor cast32。32物理目标与梯度仍有精度限制，不能由native success推断等精度。

control的Mixed32 F超额为1.49e-5–2.56e-4，Mixed64为6.17e-6–2.53e-4；最大图像差约8.8e-5噪声、core κ差RMS约2.12e-6，均满足原参考门。相比此前纯32 control的分解失败，这44个mixed尝试没有再次出现初始分解失败，支持64构建/分解在本次重复输入下解决该问题；不能由此推断所有模型/噪声都已稳定。

旧gamma flags仍单列：Full64 subhalo仅8/11通过、control 11/11；两个mixed路径各自subhalo/control均0/11。Full64全部accepted以及mixed control全部accepted是原工程场/F门的结果，不是把旧梯度结果改写为成功。此前同输入64目标/场接近而旧gamma变化的问题也仍可见。

结论限于本批固定输入：**混合精度提高了初始分解稳定性并降低耗时，但尚不能同时替代64位的目标精度和完整恢复质量要求**。主路径最接近用户指定的精度边界，64 solve控制并未提供目标精度收益；没有追加64精修、稳定能量变体或更宽阈值来改变本批结果。

最终 [结果说明](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_mixed_precision_results.md) 已只读复核：完成/accepted/旧gamma分别报告，v1预检与正式v2分开，实际速度和场误差范围与本审查的原始数组重算一致，未发现实质性不符。
