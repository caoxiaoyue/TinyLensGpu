# L-BFGS float32 / float64：精度、计时与结果核查

2026-11-03。仅主源阅读及CPU数组/metadata核验；不运行拟合、GPU或新的谱分解。标准200²、seed17、3357维，复用最初约1秒计时的truth-node源初值与零势。

## 现有精度契约与需要避免的提升

public renderer和迭代problem支持float32：[renderer](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ForwardSimulation/LensImage/potential_correction.py:109)、[公共problem](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/_potential_correction_common.py:70)。核心测试覆盖数据、坐标/权重、梯度、forward/transpose/prior及FFT的实际dtype和相反ambient x64下的作用域恢复：[precision tests](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/tests/test_potential_correction_precision.py:10)。小coupled37的支持不自动转移到原大模型；[既有精度验证](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/docs/validation/potential-correction-precision-performance.md) 明确保留大模型旧LM/单步float32失败，不能把那些结果当成本次L-BFGS结论。

原[PreparedAD.matrix](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/benchmark_scientific.py:103) 使用np.float64缓冲；ambient x64启用时，后续H0/Cholesky/白化会被提升。因此仅把图像或renderer设成32不能证明新算法为JAX32。[JAX x64规则](https://docs.jax.dev/en/latest/101/default_dtypes.html)。反之，SciPy1.17.1的L-BFGS-B本来就将x/f/g/work arrays设为float64，并强制上转用户提供的float32梯度：[已安装源码](/home/xycao/anaconda3/envs/tinylens_gpu/lib/python3.11/site-packages/scipy/optimize/_lbfgsb_py.py:442)。正确描述是“JAX数值路径float32、SciPy主机控制float64”，不应声称整个进程都是32。

## 病态先验与停止精度

原目标仍为\(F=\tfrac12(\chi^2+q^TQq)\)，势prior为\(Q_p=\lambda_4D_4^TD_4+\lambda_0I\)，完整pixel-unit stencil和\(\lambda_0=10000\)保持：[prior实现](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/utils/inversion/potential_regularization.py:66)。本审查从原956个节点独立重建完整D4对角：subhalo最大roughness diagonal=2.24e9、float32 ULP=256，幅度项仍在所有节点产生非零增量；control最大约1.4e13、ULP=1048576，912/956节点加10000后无可表示变化。未调节prior也可能在有限精度dense H0中丢失弱模信息。

F约11735时float32 ULP=0.0009765625；原ftol=1e-15对应的减小尺度远低于此。SciPy按相对F变化或投影梯度停止，[官方定义](https://docs.scipy.org/doc/scipy-1.17.0/reference/optimize.minimize-lbfgsb.html)；返回native success不能证明达到float64精度。将已舍入F再cast64不恢复信息。`highest`控制32位dot/conv精度，不将整个算法提升为64：[JAX说明](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html)。

若H0 Cholesky失败应原样保存，[JAX接口](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.cholesky.html) 明确非正定输入返回NaN。加jitter或提高幅度prior会改变本实验；用64构建/分解H0后cast或保留64则属于另行标记的mixed variant。改为分别计算平方差分范数虽可保持同数学prior，也应作为单独数值实现，而非悄悄替代本次q·Qq路径。

## 测量runner与计时边界

已只读审查 [benchmark_lbfgs_precision.py](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/benchmark_lbfgs_precision.py:1) 及 [测量时完整快照](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_precision_v1/runner_at_measurement.py)。快照SHA256为11b2ddf305563e824b401a67c56ccb3be54612fe48b04bc00d15a8c03a9ec770，与四份原始report均匹配。后续仅修正汇总/输出；原始rows未改。

TypedAD用实际point dtype构建H0，callback明确把SciPy z cast到该dtype；每fit重新计算H0和Cholesky，保留原L-BFGS参数，不用缓存数值metric、上次解或jitter。float32记录的forward、prior、AD gradient、objective、白化gradient、H0、Cholesky和最终值均为32，FFT为complex64，objective/gradient JAXPR无f64/c128。SciPy传入的host梯度64边界明确。

数据、noise、初值由同canonical64模型产生，32路径只cast；两dtype canonical input/topology hash相同。势网格使用canonical64整数拓扑和预计算浮点数组cast32，未重新构造原生32网格。这是固定canonical几何下的计算精度比较，区别于旧精度研究重新生成32几何、记录3464个query-corner变化的流程。源轴、宏射线及插值仍会舍入，不能称两者执行完全相同的浮点函数。

四个worker串行独立进程，CPU BLAS=1、最高matmul精度。cold包含fit内首次编译，warm复用kernel；计时包含数值H0、分解、优化、终点AD诊断及raw products，GPU明确同步。排除进程/CUDA启动、共同mock/几何、文件输出和独立canonical64外验；该口径是prepared reconstruction，非启动至落盘全过程。[JAX benchmark规范](https://docs.jax.dev/en/latest/benchmarking.html)。

每个完成终点用**原64 renderer/data/prior**重新计算F及图像/源/core κ，且不反馈停止。已CPU核对保存endpoint cast64与canonical values一致、输入/拓扑/配置一致，重算全部model/source/κ差、RMS和阈值flags，与 [summary](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_precision_v1/summary.json) 一致。

## 实测结果与精度分层

| 标准subhalo | Float64 | Float32 |
| --- | ---: | ---: |
| 首次fit，含首次JIT | 5.4483 s | 4.3754 s |
| 三次warm完成中位数 | 0.988013 s | 0.223298 s |
| warm接受步数 | 190–205 | 55–59 |
| native完成/尝试，含cold | 4/4 | 4/4 |
| 达既有canonical参考门 | 4/4 | 0/4 |

0.988013/0.223298≈4.42是**更低实际精度终点的观察耗时比**，不能称相同精度的成功加速。H0阶段同等工作量的warm中位数约0.4357/0.1105秒，证实有实际计算收益；优化循环同时包含较早停止/较少步骤。cold只约1.25倍。表中记录已完成运算的耗时，control中途分解失败另列。

四个subhalo32终点均有限、native ftol正常停止，主峰均在(0.75,0.075) arcsec。canonical64重算峰0.303159–0.303328，对应64峰约0.303576，差约0.08%–0.14%；图像差RMS/noise为0.000258–0.000542，源相对差0.0255%–0.0586%，core κ差RMS为0.000153–0.000276。三项场门全部通过，原有prior峰压低仍在。

但canonical64 F超额为0.001950–0.006770，四次均未达到原1e-3门，legacy gamma也未通过。不能仅因旧梯度门失败就称主形态不物理；这里有单独的、实际测得的F精度损失，且不事后抬高门。float32内部objective与64外验F不同，不能用前者替代后者。

control的64四次均完成并通过原门，warm中位数0.681965秒。32前三次完成、第四次（warm3）在初始H0 Cholesky失败；两个completed warm为0.186701/0.190541秒，中位0.188621秒，后者本身未过F门。按完整canonical门，control32总计2/4通过，warm仅1/3；完成中位数不等于合格重建中位数。失败的0.120697秒仅为中止耗时，已从完成统计排除。

## 保存矩阵支持的机制与结论

独立 [CPU谱记录](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_precision_v1/control_metric_spectra.json) 显示：64 H0最小/最大特征值约2.27152/4.98090e13，CPU double Cholesky成功；对角归一化后最小特征值仍仅8.56424e-10。失败的32 H0提升为64后，仍有-27315.90、-2838.90两个负特征值、归一化最小约-4.48185e-9，CPU double Cholesky同样失败。因此已保存的32矩阵本身丢失正定性，不能只靠提升分解运算精度修复它；完整H0需要重新在足够精度形成。

这支持病态方向与单精度舍入的解释。每次成功/失败的H0未全部归档，故未把GPU scatter累加次序认定为已隔离的根因。同输入重复4次也不是4个独立噪声场景。

最终 [结果说明](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/lbfgs_precision_results.md) 的时间、精度范围、control失败与归档修正均与原始记录一致。当前证据支持保留32位作为较快、场形态接近的研究方向；**原样纯JAX32没有同时复现64精度与稳定性，64仍应为此配置默认**。本次没有偷偷加入mixed/stable-energy变体，也未测试source-only初值或推广到其他prior/模型。
