# TinyLensGpu 安装指南

本文档提供 TinyLensGpu 的详细安装说明和依赖管理信息。

---

## 📦 安装方式

### 方式 1: 使用 pip 安装（推荐用于用户）

```bash
# 基础安装
pip install -e .

# 安装开发依赖
pip install -e ".[dev]"

# 安装所有可选依赖
pip install -e ".[all]"
```

### 方式 2: 使用 requirements.txt（推荐用于开发者）

```bash
# 安装生产依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -r requirements-dev.txt
```

### 方式 3: 使用 conda（推荐用于 GPU 环境）

```bash
# 创建新的 conda 环境
conda create -n tinylens_gpu python=3.11
conda activate tinylens_gpu

# 安装 CUDA 和 cuDNN (Arch Linux)
sudo pacman -S cuda cudnn

# 或使用 conda 安装 CUDA
conda install -c conda-forge cudatoolkit=12.0

# 安装 JAX with CUDA support
pip install -U "jax[cuda12]"

# 安装其他依赖
pip install -r requirements.txt

# 安装 TinyLensGpu (开发模式)
pip install -e .
```

---

## 🔧 依赖说明

### 核心依赖 (requirements.txt)

| 包名 | 版本要求 | 用途 |
|------|----------|------|
| jax[cuda12] | >=0.4.20 | GPU 加速数值计算 |
| caskade[jax] | >=0.1.0 | 模块化框架 |
| numpy | >=1.24.0,<2.0.0 | 数值计算 |
| scipy | >=1.10.0 | 科学计算 |
| astropy | >=5.0.0 | 天文数据处理 |
| matplotlib | >=3.5.0 | 可视化 |
| corner | >=2.2.0 | 后验分布可视化 |
| pyyaml | >=6.0 | 配置文件解析 |
| numba | >=0.57.0 | JIT 编译 |
| nautilus-sampler | >=0.6.0 | 嵌套采样 |
| dynesty | >=2.0.0 | 动态嵌套采样 |

### 开发依赖 (requirements-dev.txt)

**测试工具**:
- pytest>=7.0.0 - 测试框架
- pytest-cov>=4.0.0 - 测试覆盖率
- pytest-xdist>=3.0.0 - 并行测试

**代码质量**:
- mypy>=1.0.0 - 静态类型检查
- black>=23.0.0 - 代码格式化
- flake8>=6.0.0 - 代码检查
- isort>=5.12.0 - import 排序
- ruff>=0.1.0 - 快速 linter

**文档**:
- sphinx>=6.0.0 - 文档生成
- sphinx-rtd-theme>=1.2.0 - 文档主题

**开发工具**:
- jupyter>=1.0.0 - Jupyter notebook
- ipython>=8.10.0 - 交互式 Python

### 可选依赖

通过 `setup.py` 的 `extras_require` 安装：

```bash
# 开发工具
pip install -e ".[dev]"

# 文档生成
pip install -e ".[docs]"

# Jupyter notebooks
pip install -e ".[notebooks]"

# UltraNest 采样器
pip install -e ".[ultranest]"

# 所有可选依赖
pip install -e ".[all]"
```

---

## 🖥️ 平台特定说明

### Linux (推荐)

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# Arch Linux
sudo pacman -S python cuda cudnn

# 安装 TinyLensGpu
pip install -r requirements.txt
pip install -e .
```

### macOS

```bash
# 使用 Homebrew
brew install python@3.11

# JAX 在 macOS 上默认使用 CPU
pip install jax  # 不需要 cuda12 后缀

# 安装其他依赖
pip install -r requirements.txt
pip install -e .
```

### Windows

```bash
# 使用 Anaconda (推荐)
conda create -n tinylens_gpu python=3.11
conda activate tinylens_gpu

# 安装依赖
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 GPU 支持

### CUDA 版本

TinyLensGpu 需要 CUDA 12.x 支持。检查 CUDA 版本：

```bash
nvidia-smi
nvcc --version
```

### JAX GPU 配置

```bash
# CUDA 12.x
pip install -U "jax[cuda12]"

# 如果使用 CUDA 11.x
pip install -U "jax[cuda11_local]"

# CPU only (用于测试)
pip install jax
```

### 验证 GPU 可用性

```python
import jax
print(jax.devices())  # 应该显示 GPU 设备
```

---

## 🧪 验证安装

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_caskade_models.py

# 查看测试覆盖率
pytest --cov=TinyLensGpu --cov-report=html
```

### 快速测试

```python
import TinyLensGpu
from TinyLensGpu.Models import SIE, Shear, SersicEllipse
import jax.numpy as jnp

# 创建 SIE 模型
sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
sie.theta_E.to_static()
sie.e1.to_static()
sie.e2.to_static()
sie.center_x.to_static()
sie.center_y.to_static()

# 测试偏转角计算
x = jnp.linspace(-2, 2, 10)
y = jnp.linspace(-2, 2, 10)
X, Y = jnp.meshgrid(x, y)
alpha_x, alpha_y = sie.deriv(X, Y)

print("✅ TinyLensGpu 安装成功！")
print(f"偏转角形状: {alpha_x.shape}")
```

---

## 🔄 更新依赖

### 更新到最新版本

```bash
# 更新所有依赖
pip install --upgrade -r requirements.txt

# 更新特定包
pip install --upgrade jax[cuda12]
pip install --upgrade caskade[jax]
```

### 锁定依赖版本

```bash
# 生成精确版本的依赖文件
pip freeze > requirements-lock.txt

# 使用锁定的版本安装
pip install -r requirements-lock.txt
```

---

## 🐛 常见问题

### 问题 1: JAX CUDA 不可用

**症状**: `RuntimeError: No GPU/TPU found`

**解决方案**:
```bash
# 检查 CUDA 安装
nvidia-smi

# 重新安装 JAX with CUDA
pip uninstall jax jaxlib
pip install -U "jax[cuda12]"
```

### 问题 2: Caskade 导入错误

**症状**: `ModuleNotFoundError: No module named 'caskade'`

**解决方案**:
```bash
pip install "caskade[jax]"
```

### 问题 3: 内存不足

**症状**: `CUDA out of memory`

**解决方案**:
```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
```

### 问题 4: NumPy 版本冲突

**症状**: `numpy 2.0 is not compatible`

**解决方案**:
```bash
pip install "numpy>=1.24.0,<2.0.0"
```

---

## 📚 相关文档

- [README.md](README.md) - 项目概述
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [CASKADE_GUIDE.md](CASKADE_GUIDE.md) - Caskade 使用指南
- [TESTING.md](TESTING.md) - 测试文档
- [TYPE_ANNOTATIONS_SUMMARY.md](TYPE_ANNOTATIONS_SUMMARY.md) - 类型注解文档

---

## 💡 最佳实践

1. **使用虚拟环境**: 始终在虚拟环境中安装
2. **固定版本**: 生产环境使用 `requirements-lock.txt`
3. **定期更新**: 定期更新依赖以获取 bug 修复
4. **测试覆盖**: 更新依赖后运行完整测试套件
5. **文档同步**: 更新依赖时同步更新文档

---

**最后更新**: 2024-12-23  
**维护者**: TinyLensGpu 开发团队
