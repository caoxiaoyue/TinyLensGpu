# Lens + Source MGE Demo

## 说明

此 demo 使用 Multi-Gaussian Expansion (MGE) 来建模透镜光分布，同时建模源光分布。MGE 使用多个高斯组件来拟合复杂的光分布。

## 从 YAML 迁移到程序化 API

由于 MGE 包含大量高斯组件（通常 10-20 个），手动创建每个组件会很繁琐。建议使用循环或辅助函数来创建。

### 示例代码

```python
from TinyLensGpu.CaskadeModels import ParamU, GaussianEllipse
from TinyLensGpu.CaskadeModels.builder import build_lens_model, build_likelihood

# MGE 参数（从 YAML 或 MGE 拟合获得）
mge_sigmas = [0.01, 0.015, 0.023, ...]  # 高斯宽度
mge_weights = [0.1, 0.15, 0.12, ...]    # 相对权重

# 共享的几何参数
center_x = ParamU("center_x", 0.0, prior_type="gaussian", 
                  prior_settings=[0.0, 0.1], limits=[-3.0, 3.0])
center_y = ParamU("center_y", 0.0, prior_type="gaussian",
                  prior_settings=[0.0, 0.1], limits=[-3.0, 3.0])
e1 = ParamU("e1", 0.0, prior_type="gaussian",
            prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
e2 = ParamU("e2", 0.0, prior_type="gaussian",
            prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])

# 创建 MGE 组件列表
gaussians = []
for i, (sigma, weight) in enumerate(zip(mge_sigmas, mge_weights)):
    gauss = GaussianEllipse(
        sigma=ParamU(f"sigma_{i}", sigma),  # 固定
        center_x=center_x,  # 共享
        center_y=center_y,  # 共享
        e1=e1,  # 共享
        e2=e2,  # 共享
        flux=ParamU(f"flux_{i}", weight),  # 线性参数
    )
    gaussians.append(gauss)

# 设置动态参数
center_x.to_dynamic()
center_y.to_dynamic()
e1.to_dynamic()
e2.to_dynamic()

# 构建模型
phys_model = build_lens_model(lens_light=gaussians)

# 后续步骤与其他 demo 相同
prob_model = build_likelihood(phys_model, image_data, ...)
likelihood = VectorizedLensLikelihood(prob_model)
```

## 注意事项

1. **参数共享**: MGE 中所有高斯组件通常共享相同的中心和椭率
2. **固定宽度**: 高斯宽度 (sigma) 通常从 MGE 拟合中获得并保持固定
3. **线性参数**: 每个高斯的通量 (flux) 作为线性参数求解

## 当前状态

由于 MGE 配置复杂，建议：
1. 使用 `lens_only/run_model.py` 作为模板
2. 根据实际 MGE 参数调整代码
3. 或继续使用 YAML 配置（需要旧版本代码）

## 参考

- 主 demo 目录: `../lens_src/`
- MGE 拟合工具: 可使用 `mge_fit_1d` 或 `mge_fit_sectors` 等工具
