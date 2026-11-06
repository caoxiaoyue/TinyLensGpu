# AD + L-BFGS 与原始 LM：计时设计审查

2026-11-03。只读方法审查；本文件不运行 GPU，也不从旧实验耗时推断速度比。共同对象为原始200² mock 的 truth-node 初值 subhalo/control，float64、固定原始强度、相同 GPU 与原始物理梯度验收。

## 可比较的时间口径

主比较应是**从共同已准备几何/数据开始，到得到通过原物理梯度阈值的 MAP 重建**的 wall time。

| 新方法必须计入 | 原 dense LM 必须计入 |
| --- | --- |
| 无磁盘缓存的初始 AD GN metric | prior Q 准备与每次当前 Jacobian/GN curvature |
| 固定坐标 Cholesky、坐标变换 | 全部增量 Cholesky、拒绝/重试与阻尼控制 |
| 全部 L-BFGS/line-search 和 CPU/GPU 传输 | 全部 Python 控制、目标/梯度及同步 |
| 最终当前点 AD 梯度及 GN 对角验收 | 最终当前点重新 evaluate/prepare、梯度验收 |

绘图、NPZ/JSON 写出、真值恢复指标和与历史 LM 场比较可在两者主计时外。新方法的原 `solve_seconds` 排除了 metric、Cholesky/首编译和终点证书，不能与 `problem.fit()` 全时直接相除。[当前 scientific runner](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/run_scientific.py:54)。

原 LM 公共 `fit()` 还计算先验 normalization logdet、终点 posterior logdet/evidence 和渲染产品；可另外报告完整公共 API wall time，但不能将这些新方法没有计算的内容默认为共同 MAP 成本。[LM 初始化与迭代](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py:444)、[终点刷新与 evidence](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py:624)。

## 在不改算法的情况下区分 LM 成本

可以在实验脚本中对 `_DenseIterationBackend.initialize` 和 `.logdets` 加计时适配：

1. `initialize` 仍使用相同 `dense_prior` 与 `_logdet_spd`，保持先验 SPD 检查、返回值和终止政策。先同步 Q 已完成，再计量 `_logdet_spd(Q)`；仅这一已同步区间属于可分离的 prior-normalization 成本。不能用假 logdet 或 `positive=True` 跳过原检查。
2. 在终点 `logdets(system)` **入口**记录同步时间戳。原控制器在此之前已经刷新最终 `point`/`system`，并读取最终 scaled gradient 为 Python float，因此这个入口可以作为重建和证书完成边界。
3. `T_LM_MAP = t_logdets_entry - t_fit_start - T_prior_normalization_logdet`。这一口径保留 Q 构建、所有 LM 求解与最终当前点验收；随后真实 logdets/evidence/产品照常执行，以保留算法和公共结果。
4. 若失败而没有进入最终 `logdets`，不得报告成功 MAP 速度；保留失败状态及耗时。新方法同样必须通过外部物理证书，不能仅凭 SciPy native success 参加成功速度比。

适配应只调整计量，不能改变每一步系统矩阵、阻尼/容差、prior 或终止条件。原 owning 方法很短，便于逐行保持一致：[initialize](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py:131)、[logdets](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py:162)。

## 冷启动、warm 和异步陷阱

- **原型函数重复调用不保证 warm**：当前 `ad_metric` 每次定义新的 jitted `block`；`optimize` 每次定义新的 objective/decode/value-gradient 闭包。JIT 缓存依赖函数身份和参数签名，仅仅“调用过相同源码”不足以排除编译。建议准备并复用 numerical-kernel 对象，metric 的当前 point、strengths、indices 和 value-gradient 的 Cholesky/坐标原点作为动态数组参数。原 LM 则已保存 `_evaluate_kernel` 和 `_dense_backend.prepare` 的函数对象。官方特别提醒避免循环内重新定义临时 jitted 函数：[JIT 缓存](https://docs.jax.dev/en/latest/jit-compilation.html#jit-and-caching)。
- **warm 仍重算数值量**：每个计时重复都从同一初值重新拟合，重新构建初始 AD metric、Cholesky 和最终证书；只复用编译的程序。若复用上次已经算好的 metric 或读取磁盘缓存，需另标“metric 数值已缓存”，不能作为本次主比较。
- **冷启动标签要精确**：“solver 首编译”可以从已经准备的共同几何/数据之后起算；它不同于包含 Python import、CUDA context 和 mock 生成的全进程启动。最好每个 case/method 独立进程测首调用，再在同进程复用相同 kernel 对象测 warm。若仅 `jax.clear_caches()`，需如实标为清空 JAX 编译缓存后的 solver 首调用，不能宣称全系统冷启动。
- **扣除的 normalization 也可能热身运行时**：冷态 LM 若先执行 prior Cholesky，再减掉该整段时间，cuSOLVER/BLAS 等一次性运行时初始化可能被免费排除，而新方法首 Cholesky 仍计入。可以两者计时前做相同小矩阵 Cholesky 并同步，把 cold 明确定义为“共同 GPU/线性代数运行时已启动、solver 首次编译”；或同时呈现 cold 公共 API 全时，并把扣 normalization 的口径标为派生时间。Warm 比较没有这种一次性初始化差异。
- **同步与传输**：计时前等待输入准备完成，计时结束等待所有相关输出完成。JAX 返回 future；`block_until_ready()` 计量执行，`np.asarray` 同步并额外包含设备到主机传输。SciPy 循环必须把梯度传给 CPU，这个成本属于实际新方法，不能从新方法总量删掉。[JAX 异步说明](https://docs.jax.dev/en/latest/async_dispatch.html)、[官方 benchmark 指南](https://docs.jax.dev/en/latest/benchmarking.html)。
- **没有性能用的 callback I/O**：旧 runner 每步写 JSONL、定期压缩 checkpoint；新 benchmark 应关闭这些非求解器输出，或在时间口径中明确包含。LM 自身的内存 history 和 Python 控制是其实现成本，可保留。
- **停止质量而非步数对齐**：使用原 subhalo/control 的校准 GN 缩放梯度政策并检查每次终点；迭代预算和 SciPy 内部 gtol 不必强行相同。不得只报告数值接近但未过证书的更快终点。
- **报告分布和硬件**：保存 GPU 型号、JAX/SciPy/精度、编译政策、重复次数及每次原始 wall time；warm 用中位数并给范围。串行测量、避免其他 GPU 工作，记录顺序，不能只选最快一次。

这些条件与 JAX 官方建议一致：把编译与运行分开，保持相同 dtype，对设备执行显式同步，并明确数据传输是否进入应用时间。[Benchmarking JAX code](https://docs.jax.dev/en/latest/benchmarking.html)。

## 新 benchmark 审查状态

已只读审查 [benchmark_scientific.py](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/benchmark_scientific.py:1)：

- [PreparedAD](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/benchmark_scientific.py:46) 保存稳定的 metric/diagonal/value-gradient JIT 对象，当前点和 Cholesky 作为动态参数。每次 fit 从相同原初值开始，重建完整初始 metric、Cholesky、梯度/line search、终点当前对角证书；无磁盘 metric 缓存和优化中文件写出。
- [PreparedLM](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/benchmark_scientific.py:193) 的 initialize 保留相同 Q 和 `_logdet_spd`，在 Q 已同步后才开始排除 normalization 计时。终点 evidence 区间从真实 `.logdets` 入口到 `_result_products` 入口，保留了最终当前点/curvature/梯度刷新。
- 本脚本的主共同口径实际是 **MAP + 原始图像/场产品**：LM 总 wall time减去 prior-normalization 和 final-evidence区间；AD 总 wall time包括相同 `_result_products` 并同步。它与上文仅到证书完成的 MAP 口径略有区别，但双方一致，命名明确时可以公平比较。
- 每个 method/case 独立 worker；同一 worker 内首调用与3次重复分开记录。共同输入用 hash 检查，最后另做当前终点 AD/解析审计；软件版本、GPU、编译缓存配置和脚本 hash 均保存。

审查建议：`PreparedLM.fit` 的计时成功条件除了 `result.success`，还应检查该次 `final_scaled_gradient <= threshold`；原 API 可以从 objective/update 条件成功退出，未必自动满足本比较采用的梯度 gate。已有每次记录可直接核验此条件，不需要因为补强判定而重跑已经满足条件的样本。所有 repeat 都应通过，而非只验证最后一次独立审计。

Cold 仍需遵守前述运行时预热/标签限制，warm 可作为主要速度比较。

## 首批计时的质量门失败

首批 subhalo 已有 [LM 原始记录](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/timing_full/lm_subhalo.json) 与 [AD/L-BFGS 原始记录](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/timing_full/lbfgs_subhalo.json)。在 RTX 5080、float64、相同 input hash 下，LM 首调用与3次warm均通过原梯度门；warm MAP+products 为33.7906–33.8274秒，中位33.8023秒。

AD 首调用及前两次warm通过，但第三次warm的 SciPy native success 来自 `ftol=1e-15` 的相对函数减小，外部梯度为7.147879e-4，高于5.009761e-4阈值。该次独立解析梯度也为7.147879e-4，两者在同GN metric下差仅5.6e-14，因此这不是证书计算误差。目标函数11734.82650908051虽已非常接近其他成功终点，仍不满足已约定的验收。

该失败行0.9923秒恰是本批最短warm耗时，不能删除后只报告更漂亮的速度比，也不能把全部4行称为4次成功重建。首次批次因质量门失败而停止；随后单独补齐了control，两类原始记录均保留。

已执行只改变内部 `ftol=0` 的AD subhalo探针：[完整记录](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/timing_ftol0/lbfgs_subhalo.json)。其首调用通过，但第一次warm仍以函数相对减小返回native success，外部梯度为7.7226024e-4，高于5.009761e-4阈值，批次随即停止。解析证书确认相同结论，同GN尺度梯度差约6.6e-14。**收紧ftol没有解决本次重复运行的验收稳定性**；不能报告为已修复。双线性分支可能放大浮点轨迹差，但现有metadata不能独立证明具体失败点恰落在knot。

## 完成的速度比较与适用范围

CPU复核确认：`timing_full`四份配置均等于原始科学JSON，同case的LM/AD input hash相同，四份脚本hash均与归档 [benchmark_snapshot.py](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/timing_full/benchmark_snapshot.py) 一致；ftol0探针的独立快照hash也与其metadata一致。所有LM计时行和control AD计时行均实际通过原梯度阈值，未依赖native success代替外部验收。

| Warm、MAP+raw products | 原 dense LM 中位数 | AD成功样本耗时 | AD warm通过次数 | 仅成功样本的中位速度比 |
| --- | ---: | ---: | ---: | ---: |
| Subhalo | 33.802278 s | 1.072298–1.121123 s，中位1.096710 s | 2/3 | 30.82× |
| Control | 13.916982 s | 0.701246–0.717716 s，中位0.704305 s | 3/3 | 19.76× |

Control来源：[LM](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/timing_full/lm_control.json)、[AD/L-BFGS](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/pt_new_algorithm/timing_full/lbfgs_control.json)。Subhalo的约31倍是**满足原证书的成功运行条件下**的测量，不是稳定每次得到有效解的保证；失败耗时和ftol0失败探针必须与其共同呈现。这里的2/3和3/3只是本次小批次观察数，不是校准的长期成功率。

每次AD计时均重新计算初始AD metric、坐标Cholesky、全部优化、终点梯度/当前GN对角证书和raw products，没有复用上次metric数值或读取disk cache；只复用已编译kernel。LM排除了先验normalization和终点evidence的已分离区间，其余求解、终点刷新及raw products计入，因此上述比较不是把AD内部iteration-only时间与LM完整API时间相除。

本结论限定于RTX 5080、float64、原始200² truth-init mock、固定先验以及**当前dense LM实现**。不外推到所有LM、operator backend、其他规模或真实数据。Cold记录保留，但本批没有共同小Cholesky运行时priming，不将其比值作为严格的冷启动速度结论。无需为此次速度问题继续修改优化算法；整合前仍需解决停止/证书政策的重复运行稳定性。
