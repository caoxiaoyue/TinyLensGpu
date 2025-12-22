# TinyLensGpu 快速开始指南

## 安装

```bash
pip install -e .
```

## 最简示例

```python
from TinyLensGpu.Models import ParamU, SersicEllipse
from TinyLensGpu.Models.builder import build_lens_model, build_likelihood, load_lens_data
from TinyLensGpu.Models.prior_spec import make_prior_transformation
from TinyLensGpu.Models.likelihood import make_likelihood
from TinyLensGpu.ProbModel.Image.lens_likelihood import LensLikelihood
from nautilus import Sampler

# 1. 加载数据
image, noise, psf, _ = load_lens_data('image.fits', 'noise.fits', 'psf.fits')

# 2. 创建模型组件
sersic = SersicEllipse(
    R_sersic=ParamU("R_sersic", 1.0, prior_type="uniform", prior_settings=[0.1, 2.0]),
    n_sersic=ParamU("n_sersic", 4.0, prior_type="gaussian", prior_settings=[4.0, 0.5]),
    e1=ParamU("e1", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3]),
    e2=ParamU("e2", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3]),
    center_x=ParamU("center_x", 0.0),
    center_y=ParamU("center_y", 0.0),
    Ie=ParamU("Ie", 1.0),
)

# 3. 设置动态参数
sersic.R_sersic.to_dynamic()
sersic.n_sersic.to_dynamic()
sersic.e1.to_dynamic()
sersic.e2.to_dynamic()

# 4. 构建模型和似然
model = build_lens_model(lens_light=[sersic])
prob_model = build_likelihood(model, image, noise, psf, pixel_scale=0.074)
likelihood = LensLikelihood(prob_model)

# 5. 运行采样（与 example_v4.py 风格一致）
prior, prior_specs = make_prior_transformation(likelihood)
loglike = make_likelihood(likelihood, vectorized=True)
sampler = Sampler(prior, loglike, n_dim=len(prior_specs), n_live=200, vectorized=True)
sampler.run(verbose=True, n_eff=800)

# 6. 获取结果
samples, log_w, _ = sampler.posterior()
```

## 运行示例

```bash
cd paper/demo/lens_only
python run_model.py
```

## 核心概念

### ParamU - 带先验的参数
```python
# 均匀先验
param = ParamU("name", value, prior_type="uniform", prior_settings=[min, max])

# 高斯先验
param = ParamU("name", value, prior_type="gaussian", prior_settings=[mean, std])

# 对数均匀先验
param = ParamU("name", value, prior_type="log_uniform", prior_settings=[min, max])

# 添加硬限制
param = ParamU("name", value, prior_type="gaussian", 
               prior_settings=[mean, std], limits=[min, max])
```

### 动态 vs 静态参数
```python
# 动态参数（采样）
param.to_dynamic()

# 静态参数（固定值）
param.to_static(value)
```

### 线性参数
```python
# 光度参数可以用线性求解器优化
sersic = SersicEllipse(
    # ... 其他参数 ...
    Ie=ParamU("Ie", 1.0),  # 将通过 NNLS 求解
)

# 构建似然时启用线性求解
prob_model = build_likelihood(
    model, image, noise, psf,
    pixel_scale=0.074,
    use_linear=True,  # 启用线性求解
    solver_type='nnls'  # 使用 NNLS 求解器
)
```

## 可用模型组件

### 质量模型
```python
from TinyLensGpu.Models import SIE, Shear

# SIE 质量分布
sie = SIE(
    theta_E=ParamU("theta_E", 1.5, ...),
    e1=ParamU("e1", 0.0, ...),
    e2=ParamU("e2", 0.0, ...),
    center_x=ParamU("center_x", 0.0),
    center_y=ParamU("center_y", 0.0)
)

# 外部剪切
shear = Shear(
    gamma1=ParamU("gamma1", 0.05, ...),
    gamma2=ParamU("gamma2", 0.0, ...)
)
```

### 光分布模型
```python
from TinyLensGpu.Models import SersicEllipse, GaussianEllipse

# Sersic 轮廓
sersic = SersicEllipse(
    R_sersic=ParamU("R_sersic", 1.0, ...),
    n_sersic=ParamU("n_sersic", 4.0, ...),
    # ...
)

# 高斯轮廓
gaussian = GaussianEllipse(
    flux=ParamU("flux", 1.0, ...),
    sigma=ParamU("sigma", 0.5, ...),
    # ...
)
```

## 完整工作流程

```python
# 1. 加载数据
image, noise, psf, mask = load_lens_data(...)

# 2. 创建组件
lens_mass = [SIE(...), Shear(...)]
source_light = [SersicEllipse(...)]
lens_light = [SersicEllipse(...)]

# 3. 设置动态参数
for component in lens_mass + source_light + lens_light:
    for param in component.params:
        if should_be_dynamic(param):
            param.to_dynamic()

# 4. 构建模型
model = build_lens_model(
    lens_mass=lens_mass,
    source_light=source_light,
    lens_light=lens_light
)

# 5. 构建似然
prob_model = build_likelihood(model, image, noise, psf, pixel_scale=0.074)
likelihood = LensLikelihood(prob_model)

# 6. 提取先验
prior, prior_specs = make_prior_transformation(likelihood)
loglike = make_likelihood(likelihood, vectorized=True)

# 7. 运行采样
sampler = Sampler(prior, loglike, n_dim=len(prior_specs), ...)
sampler.run(...)

# 8. 分析结果
samples, log_w, _ = sampler.posterior()
```

## 更多信息

- 完整示例: `paper/demo/lens_only/run_model.py`
- 重构总结: `REFACTORING_SUMMARY.md`
- API 文档: `CASKADE_API.md`
