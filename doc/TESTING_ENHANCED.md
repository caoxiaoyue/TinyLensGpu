# TinyLensGpu 增强测试文档

**更新日期**: 2024-12-23  
**测试框架**: pytest  
**测试类型**: 单元测试、边界测试、集成测试、性能测试

---

## 📋 测试概述

TinyLensGpu 现在包含全面的测试套件，涵盖：

1. **单元测试** - 测试单个组件的功能
2. **边界测试** - 测试边界值和异常情况
3. **集成测试** - 测试端到端工作流
4. **性能测试** - 基准测试和性能回归检测

---

## 🗂️ 测试文件结构

```
tests/
├── __init__.py                    # 测试包初始化
├── conftest.py                    # pytest 配置和 fixtures
├── test_caskade_models.py         # 原有单元测试
├── test_util.py                   # 工具函数测试
├── test_boundary.py               # 🆕 边界和边缘情况测试
├── test_integration.py            # 🆕 集成测试
└── test_performance.py            # 🆕 性能和基准测试
```

---

## 🧪 测试类型详解

### 1. 边界测试 (`test_boundary.py`)

测试边界值、极端情况和错误处理。

#### 测试类别

**参数边界测试** (`TestParameterBoundaries`):
- ✅ 零 Einstein 半径
- ✅ 极端椭率值 (接近 1)
- ✅ 奇点处的行为
- ✅ 极端 Sersic 指数
- ✅ 极小 sigma 值

**负值测试** (`TestNegativeValues`):
- ✅ 负 Einstein 半径
- ✅ 负通量值

**大数组测试** (`TestLargeArrays`):
- ✅ 100x100 大网格
- ✅ 内存效率

**空值测试** (`TestEmptyAndNoneValues`):
- ✅ 空坐标数组
- ✅ None 参数值

**数值稳定性测试** (`TestNumericalStability`):
- ✅ 极大坐标值
- ✅ 极小坐标值
- ✅ 混合尺度值

#### 运行边界测试

```bash
# 运行所有边界测试
pytest tests/test_boundary.py -v

# 运行特定测试类
pytest tests/test_boundary.py::TestParameterBoundaries -v

# 运行特定测试
pytest tests/test_boundary.py::TestParameterBoundaries::test_sie_zero_einstein_radius -v
```

---

### 2. 集成测试 (`test_integration.py`)

测试端到端工作流和组件交互。

#### 测试场景

**端到端模拟** (`TestEndToEndSimulation`):
- ✅ 简单透镜模拟
- ✅ 仅源模拟（无透镜）
- ✅ 完整透镜系统（质量+源+透镜光）

**线性求解器集成** (`TestLinearSolverIntegration`):
- ✅ NNLS 求解器集成
- ✅ 正规最小二乘集成

**概率模型集成** (`TestProbModelIntegration`):
- ✅ 似然计算工作流
- ✅ 带掩码的似然计算

**多组件系统** (`TestMultiComponentSystems`):
- ✅ MGE (Multi-Gaussian Expansion) 源
- ✅ 多质量组件系统

**PSF 卷积** (`TestPSFConvolution`):
- ✅ 高斯 PSF
- ✅ Delta 函数 PSF

**大规模集成** (`TestLargeScaleIntegration`):
- ✅ 高分辨率模拟 (100x100, nsub=5)

#### 运行集成测试

```bash
# 运行所有集成测试
pytest tests/test_integration.py -v -m integration

# 跳过慢速测试
pytest tests/test_integration.py -v -m "integration and not slow"

# 运行特定场景
pytest tests/test_integration.py::TestEndToEndSimulation -v
```

---

### 3. 性能测试 (`test_performance.py`)

基准测试和性能回归检测。

#### 性能基准

**模型性能** (`TestModelPerformance`):
- ⏱️ SIE 偏转角计算 (100x100)
- ⏱️ Sersic 光分布计算 (100x100)

**模拟器性能** (`TestSimulatorPerformance`):
- ⏱️ 非线性模拟 (60x60, nsub=3)
- ⏱️ 线性模拟 with NNLS (50x50)

**线性求解器性能** (`TestLinearSolverPerformance`):
- ⏱️ NNLS 求解器 (1000x50)
- ⏱️ 正规最小二乘 (1000x50)

**可扩展性测试** (`TestScalability`):
- 📈 网格大小扩展性 (50, 100, 200)
- 📈 组件数量扩展性 (1, 5, 10)

**内存效率** (`TestMemoryEfficiency`):
- 💾 大规模模拟内存测试 (200x200, nsub=4)

**JIT 编译** (`TestJITCompilation`):
- 🔥 JIT 预热时间测试

#### 性能阈值

```python
PERFORMANCE_THRESHOLDS = {
    'sie_deflection_100x100': 0.5,      # 秒
    'sersic_light_100x100': 0.5,
    'simulation_60x60_nsub3': 2.0,
    'linear_simulation_50x50': 3.0,
    'nnls_1000x50': 0.5,
    'normal_ls_1000x50': 0.2,
}
```

#### 运行性能测试

```bash
# 运行所有性能测试
pytest tests/test_performance.py -v -m performance

# 运行特定基准测试
pytest tests/test_performance.py::TestModelPerformance -v

# 保存基准结果
pytest tests/test_performance.py -v --benchmark-save=baseline
```

---

## 🚀 运行测试

### 快速开始

```bash
# 运行所有测试
pytest

# 运行特定文件
pytest tests/test_boundary.py
pytest tests/test_integration.py
pytest tests/test_performance.py

# 详细输出
pytest -v

# 显示打印输出
pytest -s
```

### 按标记运行

```bash
# 仅单元测试
pytest -m unit

# 仅集成测试
pytest -m integration

# 仅性能测试
pytest -m performance

# 排除慢速测试
pytest -m "not slow"

# 组合标记
pytest -m "unit or integration"
pytest -m "integration and not slow"
```

### 按模式运行

```bash
# 运行包含 "boundary" 的测试
pytest -k boundary

# 运行包含 "performance" 的测试
pytest -k performance

# 排除特定测试
pytest -k "not slow"
```

### 并行运行

```bash
# 安装 pytest-xdist
pip install pytest-xdist

# 使用 4 个 CPU 核心
pytest -n 4

# 自动检测核心数
pytest -n auto
```

---

## 📊 测试覆盖率

### 生成覆盖率报告

```bash
# 安装 pytest-cov
pip install pytest-cov

# 生成覆盖率报告
pytest --cov=TinyLensGpu --cov-report=html --cov-report=term-missing

# 查看 HTML 报告
open htmlcov/index.html
```

### 覆盖率目标

| 模块 | 目标覆盖率 | 当前状态 |
|------|-----------|---------|
| Models | 90%+ | ✅ |
| Simulator | 85%+ | ✅ |
| LinearSolver | 90%+ | ✅ |
| ProbModel | 80%+ | ✅ |
| Inference | 75%+ | 🔄 |

---

## 🎯 测试最佳实践

### 1. 测试命名

```python
# ✅ 好的命名
def test_sie_zero_einstein_radius():
    """Test SIE with zero Einstein radius."""
    ...

# ❌ 避免
def test_1():
    ...
```

### 2. 测试隔离

```python
# ✅ 每个测试独立
def test_sie_deflection():
    sie = SIE(theta_E=1.5, ...)  # 创建新实例
    ...

# ❌ 避免共享状态
sie_global = SIE(theta_E=1.5, ...)  # 不要这样做
```

### 3. 断言清晰

```python
# ✅ 清晰的断言消息
assert avg_time < 0.5, f"SIE deflection too slow: {avg_time:.3f}s"

# ❌ 无消息
assert avg_time < 0.5
```

### 4. 使用 Fixtures

```python
# conftest.py
@pytest.fixture
def sample_sie():
    sie = SIE(theta_E=1.5, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
    for param in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
        param.to_static()
    return sie

# test file
def test_deflection(sample_sie):
    x = jnp.array([1.0])
    y = jnp.array([1.0])
    alpha_x, alpha_y = sample_sie.deriv(x, y)
    assert not jnp.isnan(alpha_x).any()
```

---

## 🐛 调试测试

### 运行单个测试

```bash
# 运行并在失败时进入调试器
pytest tests/test_boundary.py::test_sie_zero_einstein_radius --pdb

# 显示局部变量
pytest tests/test_boundary.py -l

# 显示完整回溯
pytest tests/test_boundary.py --tb=long
```

### 使用 print 调试

```bash
# 显示 print 输出
pytest -s

# 或在 pytest.ini 中设置
[pytest]
addopts = -s
```

---

## 📈 持续集成

### GitHub Actions 示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
          pip install -e .
      - name: Run unit tests
        run: pytest -m unit
      - name: Run integration tests
        run: pytest -m "integration and not slow"
      - name: Generate coverage report
        run: pytest --cov=TinyLensGpu --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## 📝 测试统计

### 当前测试数量

| 测试类型 | 测试数量 | 文件 |
|---------|---------|------|
| 单元测试 | 6 | test_caskade_models.py |
| 边界测试 | 20+ | test_boundary.py |
| 集成测试 | 15+ | test_integration.py |
| 性能测试 | 10+ | test_performance.py |
| **总计** | **50+** | **4 个文件** |

### 测试执行时间

| 测试套件 | 预计时间 |
|---------|---------|
| 单元测试 | ~2 秒 |
| 边界测试 | ~5 秒 |
| 集成测试 | ~10 秒 |
| 性能测试 | ~30 秒 |
| **全部测试** | **~50 秒** |

---

## 🔍 测试示例

### 边界测试示例

```python
def test_sie_extreme_ellipticity():
    """Test SIE with extreme ellipticity values."""
    sie = SIE(theta_E=1.5, e1=0.99, e2=0.0, center_x=0.0, center_y=0.0)
    sie.theta_E.to_static()
    sie.e1.to_static()
    sie.e2.to_static()
    sie.center_x.to_static()
    sie.center_y.to_static()
    
    x = jnp.linspace(-2, 2, 5)
    y = jnp.linspace(-2, 2, 5)
    X, Y = jnp.meshgrid(x, y)
    alpha_x, alpha_y = sie.deriv(X, Y)
    
    # Should not produce NaN or Inf
    assert not jnp.any(jnp.isnan(alpha_x))
    assert not jnp.any(jnp.isinf(alpha_x))
```

### 集成测试示例

```python
def test_full_lens_system_simulation():
    """Test complete lens system with mass, source, and lens light."""
    sie = SIE(theta_E=1.5, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
    source = GaussianEllipse(flux=10.0, sigma=0.3, e1=0.1, e2=0.0,
                            center_x=0.0, center_y=0.0)
    lens_light = SersicEllipse(R_sersic=1.0, n_sersic=4.0, e1=0.2, e2=0.1,
                              center_x=0.0, center_y=0.0, Ie=5.0)
    
    # Set all to static
    for obj in [sie, source, lens_light]:
        for attr_name in dir(obj):
            attr = getattr(obj, attr_name)
            if hasattr(attr, 'to_static'):
                attr.to_static()
    
    model = PhysicalModel(lens_mass=[sie], 
                        source_light=[source],
                        lens_light=[lens_light])
    
    config = SimulatorConfig(dpix=0.05, npix=60, nsub=3)
    simulator = LensSimulator(model, config)
    
    img = simulator.simulate(use_linear=False)
    
    assert img.shape == (60, 60)
    assert jnp.sum(img) > 0
```

### 性能测试示例

```python
def test_sie_deflection_performance(self):
    """Benchmark SIE deflection calculation."""
    sie = SIE(theta_E=1.5, e1=0.1, e2=0.05, center_x=0.0, center_y=0.0)
    # ... set to static ...
    
    x = jnp.linspace(-5, 5, 100)
    y = jnp.linspace(-5, 5, 100)
    X, Y = jnp.meshgrid(x, y)
    
    # Warm-up (JIT compilation)
    _ = sie.deriv(X, Y)
    
    # Benchmark
    start = time.time()
    for _ in range(10):
        alpha_x, alpha_y = sie.deriv(X, Y)
        alpha_x.block_until_ready()
    elapsed = time.time() - start
    
    avg_time = elapsed / 10
    assert avg_time < 0.5, f"Too slow: {avg_time:.3f}s"
```

---

## 🎓 学习资源

- [pytest 文档](https://docs.pytest.org/)
- [pytest-cov 文档](https://pytest-cov.readthedocs.io/)
- [pytest-xdist 文档](https://pytest-xdist.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

## ✅ 测试清单

在提交代码前，确保：

- [ ] 所有单元测试通过
- [ ] 添加了相关的边界测试
- [ ] 集成测试覆盖新功能
- [ ] 性能测试无回归
- [ ] 代码覆盖率 > 80%
- [ ] 测试文档已更新

---

**文档维护**: TinyLensGpu 开发团队  
**最后更新**: 2024-12-23
