# J0946 迭代 potential correction：实验记录

本记录针对实测 J0946、固定已有 M3 smooth 模型，不能套用先前低维 Mock 的
1 秒计时或恢复精度。源码入口为 `pt.py`，所有 smooth 阶段未执行。

## 数据与算法核验

从 [SLACS J0946 工作目录](../slacs0946_potential_correction/) 复制了当时的
`macro_model.py`、全部 `macro/`、`data/`，共 37 个文件、
154796984 bytes，逐文件 SHA256 一致。Handoff NPZ SHA256 为
`d09a1102a64f74dcf540530d850ed633430313009d44bc0f5b5b551e18b10bda`。
保留 2293 data pixels、10000 source、1134 potential / 802 derivative-valid nodes。
Source adaptive weights 固定，完整 coupled evidence，无 block determinant 替代。

真实数据预检的白化二次型误差 7.8e-16，独立 augmented-QR 与 data-Cholesky
logdet 差 5.5e-12；JAX AD 与公开 analytical kernel 的 objective 差 0、gradient
相对差 4.3e-13。独立审查还从保存真实数组重构 penalty，差 2.3e-13。
这些说明目标函数/梯度/矩阵接口正确，不代表每个超参数候选都已收敛。
首轮预检的 data-Gram 中位数尚未过滤零响应；正式探索前已修正为旧单步脚本的
positive-response 规则。旧预检仅保留用于诊断，没有混入正式搜索的评分。

## 预检暴露的限制

| 测试 | 终点情况 | 处理 |
|---|---|---|
| 保存 M3 source 直接初始化，首轮中点 | 原生 SciPy success，GN proxy≈0.105 | 不合格，未评分 |
| 按修正后经验 bounds 的中点 | 达 3000 步，proxy≈0.0635 | 不合格，未评分 |
| 接近旧 single-step 强度的数值检查点 | 原生 success，proxy≈0.0107 | 不合格，未评分 |
| 每 λs 先 exact conditional source，中点 | 达 3000 步，proxy≈0.0362 | 不合格，未评分 |
| 强 prior 数值 anchor | 23 步，proxy≈3.17e-8 | 合格 |

原始 source 插值在 bbox 外硬截零。保存失败点的白化下降方向上，步长1e-5
预测 ΔF≈-3.9e-6，实际 ΔF≈+0.03129；缩小到1e-8才重新匹配导数。
最接近 bbox 的光线几乎落在边界。重置 L-BFGS 记忆、ftol=0 的三次诊断重启
没有解决这个停止点，不能将其归因于 Cholesky 错误，或仅凭 SciPy success 接纳。

这里的 proxy 是局部 GN 二次模型诊断，不是真实 gap 的上界。硬截断在边界
source 非零时甚至不连续，光滑优化器的梯度证书本身不足以解决跨边界问题。
本实验保留原插值/先验/验收阈值；没有为了得到想要的图而平滑边界、扩展网格或
投影/重标定 source/potential，也没有用文献中预期 subhalo 位置挑解。

## DIRECT + Powell 正式探索

```bash
python workspace/iteration/pt.py \
  --validation-log10 -3.4849016084 7.5 6.5 \
  --direct-maxfun 60 --local-maxfun 60 \
  --output workspace/iteration/direct_powell
```

每候选默认先在零 potential 下精确求 source conditional MAP，再做联合 L-BFGS。
每组初值只依赖该组 λs 与固定数据，彼此独立；预条件参考点仍为保存的 M3 source。
强 prior anchor 是已声明的数值检查点，不是从旧 potential/source 解热启动。
外层搜索范围从本次 Float64 数据响应按旧单步规则确定：

- log10 λs：[-5.48490,-1.48490]
- log10 λ4：[-1.20519,7.79481]
- log10 λ0：[0.93782,6.93782]

完整探索已执行。193 次 DIRECT 评估之后运行 Powell；后者60次函数调用中有1次
命中缓存、59次实际新拟合。另含1次数值 anchor、1次强制最佳点复跑，共 **254次
内层拟合、253个不同超参数点**。36次合格、218次未通过；其中75次原生迭代预算
耗尽，143次原生 SciPy success 但未过 GN proxy，未出现 factor exception。

| 项目 | 结果 |
|---|---:|
| 最佳单次合格候选 | Powell 访问过的 #236 |
| λs | 1.0843505375e-4 |
| λ4 | 7.3782954475e5 |
| λ0 | 1.5705034826e6 |
| local-GN log-evidence | 6240.464467834 |
| 相对数值 anchor 的 log-evidence 提升 | 29.396870354 |
| χ² / data pixel | 0.439256638 |
| 标准化 residual RMS | 0.66276439 |
| δκ 正峰及位置 | 0.043664736 @ (−0.7″,1.0″) |
| δκ 最小值 / core RMS | −0.07191279 / 0.00995950 |

上述是**已评估且通过检查的候选中最高分**，不是已确认的全域最优。DIRECT 本身
找到的最高合格 log-evidence 仅5970.83778，未胜过外部 anchor；Powell 从 anchor
开始后改善到了 #236。两个外层优化器均因预算退出。Powell 原生返回的最终端点
没有通过内层检查，保留的是它曾访问过的合格候选，而非该失败端点。

### 所选参数的三次恢复

| 尝试 | 原生 SciPy | GN proxy（阈值1e-3） | 诊断 log-evidence | δκ 正峰 |
|---|---|---:|---:|---:|
| 搜索内 #236 | success | 0.000667608 | 6240.464467834 | 0.043664736 |
| 原进程冷启动复跑 #253 | success | **0.001134934：未通过** | 6240.460312349 | 0.043665385 |
| 新进程独立验证 | success | 0.000935254 | 6240.467861399 | 0.043665076 |

三个 κ 峰都在 (−0.7″,1.0″)。相对 #236，两次复跑的 joint 解相对差分别约
1.36e-5、1.52e-5，κ core 最大绝对差分别4.25e-6、4.47e-6；objective 差小于
9.1e-7，log-evidence 最大差约0.0042。**恢复场实际上很接近，但预先声明的严格
数值门槛没有三次全过**，因此保留 `selected_repeat_passed=false`，不放宽门槛，
也不以第三次通过替换第二次未通过。两次复跑的白化、独立 QR determinant 和公开
analytical gradient 审计均通过。单次数值合格、物理解接近、完整参数域稳健恢复
是不同层面的结论；本次尚未确认后者。

### 耗时与退出记录

搜索阶段约 **52.54分钟**（protocol 时间戳至最后 trace；未含最初模型装配/导入），
准备约2.64秒。最佳候选 #236 的一次完整评估3.05秒，其中白化/初始化0.16秒、
L-BFGS2.29秒、终点 evidence0.60秒；一些未通过的候选跑满3000步，约25–40秒。
这与早先小规模 Mock 的暖启动1秒不属同一规模/计时范围。

DIRECT 的 `maxfun=60` 是软预算，本轮实际193次；这种超出不能叫全局收敛。
最佳点复跑失败触发原实验脚本的预定异常，早于原生结果和总计时的序列化。
随后仅在 CPU 用已记录分数确定性 replay：193个 DIRECT 查询及60个 Powell
调用与 trace 逐点完全一致，恢复两者的原生 status=1、预算退出信息，**没有新增
任何模型评估**。记录在 `replayed_outer_status.json`，未知坐标会立即拒绝 replay。
补写的 `summary.json` 明确标记 `repeat_gate_failed`；52.54分钟为时间戳估计，
不伪装成原进程丢失的 monotonic 总计时。当前入口已改为先持久化外层状态，并在
复跑未过时先保存完整 summary/failure，再显式报错。

当前入口的默认强 prior 检查点亦在新进程通过：24步、proxy4.33e-7，logE6203.7234。
完整探索采用的是上述显式60+60配置，不宣称已经运行默认120+80的大预算搜索。

每次运行的 `source_snapshot/` 与 `protocol.json` 冻结实际执行版本与依赖 hash。
探索版最初导出的 `source_initial`/部分 provenance 使用 M3 reference 命名；实际
内层起点以每候选 `source_initialization=conditional_source` 和 protocol.cold_start
为准。原导出归档在 `direct_powell/original_export/`，已按相同 λ/data 的新进程
捕获初值补正标签，初始 objective 差仅2.3e-13。`export_provenance.json` 记录前后
hash，所选 q、κ 与 evidence **均未改变**。不可变 trace/protocol/source_snapshot
仍保留原版。当前入口直接记录真实初始向量，并分别导出它与 `source_reference_M3`。

## 图与结论

[查看最终候选图](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/iteration/direct_powell/maximum_evidence/diagnostics.png)，
[PDF](/home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/workspace/iteration/direct_powell/maximum_evidence/diagnostics.pdf)。图中显式标记复跑检查未全部
通过；数据、模型、residual、source 与 δκ 来自保存的 #236，未选择后一次更好看的
重建替换。所有37个原始/复制 smooth 文件的 SHA256 最终复查仍一致。

**实现已完成并得到接近可重复的北侧 κ 修正峰；完整超参数域的稳定优化尚未过关。**
下一步应先检查 source 网格覆盖和边界插值的连续性，再评估更完整的超参数优化。
实测数据没有 injected 真值，当前约0.044的峰不等于已确认的 subhalo 质量或检测
显著性，也不能直接与不同模型定义的旧单步 evidence 做 Bayes factor。

## 回归测试

在 `examples/tests/`、独立 CPU JAX 进程运行：

```bash
JAX_PLATFORMS=cpu pytest test_j0946_iteration.py -q
```

4 项通过。覆盖 complete-coupling determinant、初始白化/chain gradient/GN proxy、
conditional-source 解与直接 normal equations 对比，以及缺失/坏 checksum/非正
adaptive scale 不会自动触发 smooth fitting。

最终图已逐格目视检查，plot-area alignment 0 fail/0 warn，PDF 字体最小8pt，
rendered collision 0 fail/0 warn。源码预检0 fail；300dpi PNG、报告宽度和未出TIFF
这3个投稿规格提醒按报告用途保留，见 `figure_contract.md`，不称投稿成品。
