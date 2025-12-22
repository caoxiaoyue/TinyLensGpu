# TinyLensGpu Demo 目录

本目录包含多个引力透镜建模的示例，展示如何使用 TinyLensGpu 的程序化 API。

## 目录结构

- **lens_only/** - 仅透镜光分布建模
- **src_only/** - 仅源光分布建模（含质量）
- **src_only_poslike/** - 源光分布建模 + 位置似然约束
- **lens_src/** - 透镜光 + 源光联合建模
- **lens_only_mge/** - 使用 MGE 的透镜光建模
- **lens_src_mge/** - 使用 MGE 的透镜光 + 源光建模

## 使用方法

### 新的程序化 API（推荐）

每个目录现在都包含 `run_model.py`，使用最新的程序化 API：

```bash
cd lens_only
python run_model.py
```

### 主要特点

1. **无需 YAML 配置** - 直接在 Python 代码中定义模型
2. **使用 ParamU** - 参数自动包含先验信息
3. **JAX vmap 加速** - 批处理性能提升 10-100x
4. **类型安全** - 完整的类型提示

### 示例代码结构

```python
from TinyLensGpu.Models import ParamU, SersicEllipse
from TinyLensGpu.Models.builder import build_lens_model, build_likelihood
from TinyLensGpu.ProbModel.Image import VectorizedLensLikelihood

# 1. 创建组件
sersic = SersicEllipse(
    R_sersic=ParamU("R_sersic", 1.0, 
                    prior_type="uniform", 
                    prior_settings=[0.0, 2.0]),
    n_sersic=ParamU("n_sersic", 4.0,
                    prior_type="gaussian",
                    prior_settings=[4.0, 0.5]),
    # ...
)

# 2. 构建模型
phys_model = build_lens_model(lens_light=[sersic])

# 3. 设置动态参数
sersic.R_sersic.to_dynamic()
sersic.n_sersic.to_dynamic()

# 4. 构建 likelihood
prob_model = build_likelihood(phys_model, image_data, ...)
likelihood = VectorizedLensLikelihood(prob_model)

# 5. 运行采样
from nautilus import Sampler
sampler = Sampler(prior, loglike, vectorized=True)
sampler.run()
```

## 旧的 YAML 配置（已弃用）

旧的 `run_model_from_yaml.py` 脚本已不再维护。请使用新的 `run_model.py`。

如果需要从 YAML 迁移，参考各目录中的 `run_model.py` 示例。

## 输出

每个 demo 运行后会在 `output/` 目录生成：

- `result_samples.csv` - 后验样本
- `result_summary.csv` - 后验统计摘要
- `results.pkl.gz` - 完整结果（pickle 格式）

## 性能提示

1. **批大小**: 增加 `n_batch` 可提高性能（如 `n_batch=128`）
2. **GPU 加速**: 设置 `JAX_PLATFORM_NAME=gpu` 使用 GPU
3. **并行化**: JAX 自动利用多核 CPU

## 常见问题

### Q: 如何从 YAML 配置迁移？

A: 参考对应目录的 `run_model.py`，将 YAML 中的参数转换为 `ParamU` 对象。

### Q: 如何添加新的模型组件？

A: 使用 `ParamU` 创建参数，然后用 `build_lens_model` 组合：

```python
from TinyLensGpu.Models.mass import SIE, Shear

sie = SIE(theta_E=ParamU("theta_E", 1.5, ...))
shear = Shear(gamma1=ParamU("gamma1", 0.0, ...))

model = build_lens_model(lens_mass=[sie, shear])
```

### Q: 如何调试模型？

A: 使用小批量测试：

```python
# 测试单次 likelihood 评估
test_theta = prior(np.array([0.5, 0.5, 0.5, 0.5]))
test_loglike = loglike(test_theta)
print(f"Test log-likelihood: {test_loglike}")
```

## 更多信息

- **完整文档**: 参考 `../../QUICKSTART.md`
- **API 参考**: 参考 `../../VECTORIZED_LIKELIHOOD_GUIDE.md`
- **重构说明**: 参考 `../../REFACTORING_SUMMARY.md`

---

**更新日期**: 2025-12-22  
**版本**: 2.0 (程序化 API)
