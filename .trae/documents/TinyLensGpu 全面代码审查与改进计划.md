## 约束确认（按你的最新指示）

* **忽略** 1) 依赖与安装策略重构。

* <br />

  1. 线性求解与证据计算：**只做**（a）`slogdet` 的 `sign` 健壮性处理（非正定→`-inf`），（b）`Sigma` 计算可选且默认关闭。

* <br />

  1. PSF：做“复杂度优化”，但**不**给 `PixelizedLensSimulator` 增加 sparse PSF 选项/自动阈值切换。

* 除上述限制外，其余审查项 **默认执行**。

## 将执行的改动

### 2) 证据计算健壮性（严格只做两项）

* **`slogdet`** **sign 检查**：在 [LinearInversion](file:///Users/xycao/workspace/lens_model_jax_github/TinyLensGpu/TinyLensGpu/utils/inversion/linear_solver.py) 的 `_precompute_terms(...)` 中，对 N（full matrix 分支）、H、M 的 `sign` 做检查；任一非正定/非有限即让 `log_evidence()` 返回 `-jnp.inf`。

* **`Sigma`** **默认不算**：调整 `LinearInversion.invert()` 为 `invert(return_cov: bool = False)`；默认仅返回 `s`，需要协方差时显式开启。同步更新 [reconstruct\_source](file:///Users/xycao/workspace/lens_model_jax_github/TinyLensGpu/TinyLensGpu/ObservationModel/LensImage/pixelized_image_model.py#L279-L302) 的调用以不再触发求逆。

### 3) dtype/精度策略统一（不触碰 LinearInversion 的 float32 强制策略）

* 在推断包装层把“强制 float32”改为“跟随输入 dtype 或可配置 dtype”，避免无意降低精度：

  * [make\_likelihood](file:///Users/xycao/workspace/lens_model_jax_github/TinyLensGpu/TinyLensGpu/Inference/build_likelihood.py#L14-L81)：不再无条件 `jnp.float32`，并确保返回类型满足采样器（标量/批量）。

  * [make\_prior\_transformation](file:///Users/xycao/workspace/lens_model_jax_github/TinyLensGpu/TinyLensGpu/Inference/build_prior.py#L140-L171)：避免强制 float32；保持 prior 变换数值稳定（clip 仍保留）。

### 4) 像素化源：缓存与批量评估边界清晰化

* 重构 [PixelizedImageProbModel.\_get\_or\_build\_inverter](file:///Users/xycao/workspace/lens_model_jax_github/TinyLensGpu/TinyLensGpu/ObservationModel/LensImage/pixelized_image_model.py#L171-L240) 的缓存 key：

  * 避免遍历 `vars(mass_comp)` + `float(...)` 造成的 device 同步与潜在 tracer 问题。

  * 改为“显式、可维护”的参数提取（例如按组件类型/参数列表顺序收集），并尽量在 host 侧以不可变结构表示。

* 明确像素化证据是否支持向量化：若当前实现不适合 `vectorized=True`，则在接口层给出明确报错/文档说明（避免悄悄退化性能或隐式同步）。

### 5) PSF 矩阵构建复杂度优化（不改 PixelizedLensSimulator 的 sparse 策略）

* 优化 [build\_psf\_matrix\_dense](file:///Users/xycao/workspace/lens_model_jax_github/TinyLensGpu/TinyLensGpu/utils/lensing/psf.py#L16-L94)：

  * 从 O(n\_valid^2) 双循环改为 O(n\_valid \* |PSF|)：为每个有效像素仅遍历 kernel 支撑并通过“(h,w)→index”查表定位列索引。

  * 输出仍为 dense matrix，保持 `PixelizedLensSimulator` 现有调用方式不变。

* 如 `build_psf_matrix_sparse()` 现存，将同步降构建复杂度（不新增“PixelizedLensSimulator 自动切换到 sparse”的功能）。

### 6) 文档一致性与测试补齐

* 修订 `doc/GUIDE.md` 中与当前代码不一致的入口描述（以 programmatic demo 为准；YAML runner 相关内容改为“暂未实现/计划项”）。

* 增补/更新测试：

  * `LinearInversion.log_evidence()` 非正定返回 `-inf` 的回归用例。

  * `invert(return_cov=False)` 不求逆且 API 与 `reconstruct_source()` 匹配。

  * PSF dense matrix 新旧实现（小尺寸）数值一致性测试。

  * 像素化缓存 key 在参数变化/不变化时的命中行为测试。

  <br />

## 验证方式（你确认后我会执行）

* 运行 pytest（至少覆盖 integration + pixelized + boundary 相关用例）。

* 对关键改动补充最小回归测试，保证数值与行为不退化。

## 明确不做的项

* 不做依赖/安装策略重构。

* 不做 `PixelizedLensSimulator` sparse PSF 选项或自动切换。

