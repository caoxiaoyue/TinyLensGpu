# Caskade Native Parameter Linking

**Date**: 2025-12-17
**Update**: Changed from `object.__setattr__` to caskade's native `setattr` for parameter linking

---

## Summary

使用 **caskade 原生的参数链接机制** (`setattr`) 而不是 `object.__setattr__`，这更符合caskade的设计理念。

---

## Caskade 参数链接机制

### 原理

当你在caskade Module中进行参数赋值时：
```python
module2.param = module1.param
```

caskade会创建一个**内部链接**，使得两个模块共享同一个参数值。这是caskade原生支持的参数共享机制。

### 测试验证

```python
import caskade as ck

class TestModule(ck.Module):
    def __init__(self, param_a=None):
        super().__init__()
        self.param_a = ck.Param('param_a', param_a)

m1 = TestModule(param_a=1.0)
m2 = TestModule(param_a=2.0)

# 使用caskade原生的参数链接
m2.param_a = m1.param_a  # 或者 setattr(m2, 'param_a', m1.param_a)

# 修改 m1 的值
m1.param_a.to_static(99.0)

# m2 会自动同步
print(m2.param_a.value)  # 输出: 99.0 ✓
```

---

## 两种方法的比较

### 方法1: `setattr()` (caskade原生) ✅ 推荐

```python
setattr(target_module, param_name, source_param_obj)
```

**优点**:
- ✅ 使用caskade原生机制
- ✅ 符合caskade设计理念
- ✅ 功能完全正常（值共享）
- ✅ 代码更清晰、更易理解

**缺点**:
- ⚠️ 对象身份检查 `is` 返回 `False`（但不影响功能）

### 方法2: `object.__setattr__()` (直接赋值)

```python
object.__setattr__(target_module, param_name, source_param_obj)
```

**优点**:
- ✅ 对象身份检查 `is` 返回 `True`
- ✅ 完全相同的对象

**缺点**:
- ⚠️ 绕过caskade的内部机制
- ⚠️ 可能与caskade未来版本不兼容
- ⚠️ 不符合caskade设计理念

---

## 实现细节

### config_parser.py 中的参数链接

```python
def _apply_parameter_links(self, lens_mass, source_light, lens_light):
    """
    Apply parameter links using caskade's native pointer mechanism.

    When you assign a Param from one module to another (e.g., `module2.param = module1.param`),
    caskade creates an internal link so that both modules share the same parameter value.
    This is caskade's native way of implementing parameter sharing.
    """
    # ...

    # 使用caskade原生的参数链接
    setattr(target_module, param_name, source_param_obj)
```

**为什么这样做**:
- 使用caskade原生的 `setattr` 进行参数链接
- caskade内部会创建链接，使参数值同步
- 符合caskade的设计模式

### composite.py 中的列表存储

```python
def __init__(self, lens_mass=None, source_light=None, lens_light=None):
    super().__init__()

    # IMPORTANT: 这里必须使用 object.__setattr__
    # 因为普通赋值会触发caskade的NodeList转换，导致GraphError
    object.__setattr__(self, '_lens_mass_list', lens_mass or [])
    object.__setattr__(self, '_source_light_list', source_light or [])
    object.__setattr__(self, '_lens_light_list', lens_light or [])

    # 注册每个模块（使用普通的setattr）
    for i, mass in enumerate(self._lens_mass_list):
        setattr(self, f"lens_mass_{i}", mass)
```

**为什么这里需要 `object.__setattr__`**:
- 避免caskade将列表转换为 `NodeList`
- `NodeList` 要求所有子节点有唯一名称
- MGE模型有15个同名的 "GaussianEllipse"，会导致 `GraphError`
- 这是特殊情况，必须绕过caskade的拦截

---

## 测试结果

### 功能测试 ✅

```bash
# 参数链接功能测试
python -c "
from TinyLensGpu.CaskadeInference.config_parser import CaskadeConfigParser
parser = CaskadeConfigParser('model_config.yaml')
lens_light = parser.phys_model.lens_light

# 修改 Gaussian 0 的 center_x
lens_light[0].center_x.to_static(7.77)

# 检查其他 Gaussians 是否同步
assert lens_light[1].center_x.value == 7.77
assert lens_light[14].center_x.value == 7.77
"
```

**结果**: ✅ 所有参数正确链接和同步

### 完整测试套件 ✅

```bash
pytest tests/test_caskade_models.py \
       tests/test_config_parser.py \
       tests/test_caskade_inference.py
```

**结果**: ✅ **20/20 tests passed** in 3.49s

---

## MGE 参数链接示例

### 配置文件 (lens_only_mge)

```yaml
lens_light_list:
  - type: Gaussian  # Gaussian 0
    params:
      center_x:
        fixed: false
        prior_type: gaussian
        prior_settings: [0.0, 0.1]
      # ... 其他参数

  - type: Gaussian  # Gaussian 1
    params:
      center_x:
        fixed: true
        fixed_value:
          component_type: lens_light_list
          component_idx: 0
          parameter: center_x
      # ... 其他参数 (center_y, e1, e2 也链接到 Gaussian 0)

  # Gaussians 2-14 类似
```

### 参数链接结果

- **56个参数链接**: 14个Gaussians × 4个共享参数 (center_x, center_y, e1, e2)
- **4个动态参数**: 只采样 Gaussian 0 的 center_x, center_y, e1, e2
- **15个线性参数**: 每个Gaussian的振幅通过NNLS求解
- **参数减少**: 从90个 → 34个独立参数（节省62%）

### 运行时行为

```python
# 在采样过程中，只需要设置 Gaussian 0 的参数
lens_light[0].center_x.to_static(new_value)

# 其他14个 Gaussians 自动同步
assert all(g.center_x.value == new_value for g in lens_light)
```

---

## 总结

### 最佳实践

1. **参数链接**: 使用caskade原生的 `setattr()` ✅
   ```python
   setattr(target_module, param_name, source_param_obj)
   ```

2. **列表存储**: 在 `PhysicalModel.__init__` 中使用 `object.__setattr__()` ✅
   ```python
   object.__setattr__(self, '_lens_mass_list', lens_mass or [])
   ```

### 为什么这样设计

- **参数链接**: 使用caskade原生机制更符合框架设计，代码更清晰
- **列表存储**: 必须绕过NodeList转换才能支持多个同名模块（MGE）

### 测试覆盖

- ✅ 单元测试: 20/20 通过
- ✅ MGE测试: 56个参数链接正确工作
- ✅ 完整工作流: lens_src demo通过
- ✅ 向后兼容: 所有现有配置文件正常工作

---

**结论**: 现在使用caskade原生的参数链接机制，代码更清晰、更符合框架设计理念，同时保持完全的功能性。
