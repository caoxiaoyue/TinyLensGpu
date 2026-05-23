#!/usr/bin/env python

from setuptools import setup, find_packages

# Read the long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies
install_requires = [
    "jax[cuda12]>=0.4.20",
    "caskade[jax]>=1.0.0",
    "numpy>=1.24.0,<2.0.0",
    "scipy>=1.10.0",
    "astropy>=5.0.0",
    "matplotlib>=3.5.0",
    "corner>=2.2.0",
    "pyyaml>=6.0",
    "numba>=0.57.0",
    "nautilus-sampler>=0.6.0",
    "dynesty>=2.0.0",
    "jaxnnls>=1.0.0",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-xdist>=3.0.0",
        "mypy>=1.0.0",
        "types-PyYAML>=6.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "isort>=5.12.0",
        "ruff>=0.1.0",
        "pre-commit>=3.0.0",
    ],
    "docs": [
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "sphinx-autodoc-typehints>=1.22.0",
    ],
    "notebooks": [
        "jupyter>=1.0.0",
        "ipykernel>=6.20.0",
        "notebook>=6.5.0",
        "ipython>=8.10.0",
    ],
    "ultranest": [
        "ultranest>=3.5.0",
    ],
}

# Add 'all' option to install everything
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name='TinyLensGpu',
    version='0.1.0',
    description='GPU-accelerated gravitational lens modeling with JAX',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Xiaoyue Cao',
    author_email='cxylzlx@gmail.com',
    url='https://github.com/caoxiaoyue/TinyLensGpu',
    project_urls={
        "Bug Tracker": "https://github.com/caoxiaoyue/TinyLensGpu/issues",
        "Documentation": "https://github.com/caoxiaoyue/TinyLensGpu",
        "Source Code": "https://github.com/caoxiaoyue/TinyLensGpu",
    },
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords="gravitational lensing, astronomy, GPU, JAX, nested sampling",
)
