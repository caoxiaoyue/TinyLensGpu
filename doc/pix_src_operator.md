# Pixelized Source Model: Operator Backend Implementation Notes

This document provides a detailed explanation of the implementation principles behind the **operator backend** (matrix-free backend) for pixelized sources in TinyLensGpu. The core idea of this backend is to avoid explicitly constructing the massive linear mapping matrix $A$. Instead, it defines "Linear Operators" to compute $x \mapsto Ax$ and $y \mapsto A^T y$, thereby significantly saving memory and utilizing FFT to accelerate convolution.

If you are already familiar with the matrix backend (i.e., explicitly constructing $F$/$A$ matrices and performing linear algebra), the key to understanding the operator backend lies in:

1.  **Mathematical essence remains unchanged**: We are still solving the same semi-linear inversion problem.
2.  **Computation method changes**: The large matrix $A$ is treated as a black-box function that only provides interfaces for "forward multiplication" and "transpose multiplication," without storing the matrix itself.

---

## 1. Core Code Index

Understanding the implementation of the operator backend primarily involves the following files:

-   **Solver Entry Point**: [operator_solver.py](file:///home/cao/data_disk/tinylens_dev/TinyLensGpu/TinyLensGpu/utils/inversion/operator_solver.py) (Implements CG, FISTA, SLQ)
-   **Model Top-Level**: [pixelized.py](file:///home/cao/data_disk/tinylens_dev/TinyLensGpu/TinyLensGpu/ForwardSimulation/LensImage/pixelized.py) (Responsible for assembling the Operator)
-   **Mapping Strategy**: [mapping_strategies.py](file:///home/cao/data_disk/tinylens_dev/TinyLensGpu/TinyLensGpu/ForwardSimulation/LensImage/pixelized_core/mapping_strategies.py) (Defines the $M$ operator)
-   **Assembly Factory**: [inversion_assembler.py](file:///home/cao/data_disk/tinylens_dev/TinyLensGpu/TinyLensGpu/ForwardSimulation/LensImage/pixelized_core/inversion_assembler.py) (Connects the model with the solver)

---

## 2. Linear Model Review

### 2.1 Basic Equations

We aim to recover the source plane pixel coefficients $s$ from the observed data $d$.

-   $d \in \mathbb{R}^{n_{\mathrm{data}}}$: The 1D vector of flattened unmasked image pixels.
-   $N = \mathrm{diag}(\sigma^2) \in \mathbb{R}^{n_{\mathrm{data}}\times n_{\mathrm{data}}}$: Noise covariance matrix (diagonal).
-   $s \in \mathbb{R}^{n_{\mathrm{src}}}$: The source plane coefficients to be solved.

The forward model is expressed as:

$$
d \approx A s + n, \quad n \sim \mathcal{N}(0, N)
$$

where the Blurred Lens Mapping Matrix $A$ is decomposed into two steps:

$$
A = P \, M
$$

1.  **Mapping Operator $M$** (Mapping): Maps source plane coefficients $s$ to the image plane (unblurred). This is essentially an interpolation operation (Geometric Lensing + Interpolation).
2.  **PSF Operator $P$** (PSF Convolution): Convolves the light distribution on the image plane with the Point Spread Function (PSF).

### 2.2 Bayesian Inversion

Introducing a Gaussian prior (regularization):

$$
p(s) \propto \exp\left(-\frac{1}{2} s^T H s\right)
$$

Maximum A Posteriori (MAP) estimation is equivalent to minimizing the following objective function:

$$
\mathcal{L}(s) = \frac{1}{2} (d - As)^T N^{-1} (d - As) + \frac{1}{2} s^T H s
$$

Taking the derivative with respect to $s$ and setting it to 0 yields the Normal Equations:

$$
(A^T N^{-1} A + H) s = A^T N^{-1} d
$$

The Matrix backend explicitly computes $F = A^T N^{-1} A + H$ and solves it; whereas the **Operator backend** uses iterative methods (such as Conjugate Gradient) to solve this linear system, which only requires computing Matrix-Vector Products (MVP).

---

## 3. Operator Implementation: Turning Matrices into Functions

The core of the Operator backend is how to efficiently implement $x \mapsto Ax$ (Forward) and $y \mapsto A^T y$ (Adjoint).

### 3.1 Mapping Operator $M$ (Mapping)

$M$ is a sparse operator. For each image pixel $i$, it relates to only a few nodes on the source plane (e.g., 4 points for bilinear interpolation, $k$ points for kNN).

**Data Structure**:
Instead of storing a sparse matrix object, it stores two compact arrays:
-   `weights`: shape `(n_data, k)`, storing interpolation weights.
-   `indices`: shape `(n_data, k)`, storing corresponding source node indices.

**Forward Operation (Forward)** $d = Ms$:
$$
d_i = \sum_{m=1}^{k} w_{i,m} \, s_{\text{idx}_{i,m}}
$$
Code corresponds to `operator_solver._apply_mapping`, implemented using `jax.numpy.take` and summation.

**Adjoint Operation (Adjoint/Transpose)** $s = M^T d$:
$$
s_j = \sum_{i} \sum_{m: \text{idx}_{i,m}=j} w_{i,m} \, d_i
$$
This is a "Scatter-Add" operation. Code corresponds to `operator_solver._apply_mapping_transpose`, implemented using `out.at[indices].add(...)`.

**Construction Sources**:
-   Regular Grid: [RectBilinearMappingStrategy](file:///home/cao/data_disk/tinylens_dev/TinyLensGpu/TinyLensGpu/ForwardSimulation/LensImage/pixelized_core/mapping_strategies.py)
-   Irregular Grid: [KnnKernelMappingStrategy](file:///home/cao/data_disk/tinylens_dev/TinyLensGpu/TinyLensGpu/ForwardSimulation/LensImage/pixelized_core/mapping_strategies.py)

### 3.2 PSF Operator $P$ (Convolution)

$P$ is a dense operator but has a convolutional structure. If written explicitly as a matrix, it would be a huge Toeplitz matrix. The Operator backend uses FFT to implement efficient convolution.

**Operation Flow**:

Since the data vector $d$ is unmasked (possibly irregular shape), while FFT requires a regular grid, two auxiliary operators are defined:
-   $S$ (Scatter): Fills unmasked data into the full 2D image `(height, width)`, padding masked areas with 0.
-   $G$ (Gather): Extracts unmasked pixels from the full 2D image.
-   $C$ (Convolution): Performs FFT convolution on the 2D grid.

**Forward Operation (Forward)** $P$:
$$
P = G \, C \, S
$$
Code corresponds to `operator_solver._apply_psf_unmasked_to_unmasked(..., adjoint=False)`.
1.  **Scatter**: `image[mask] = x`
2.  **FFT Convolution**:
    -   To simulate non-cyclic linear convolution, FFT needs padding to `(h + psf_h - 1, w + psf_w - 1)`.
    -   Compute `ifft(fft(image) * fft(psf))`.
    -   Crop the result back to the original size.
3.  **Gather**: Extract `image[mask]`.

**Adjoint Operation (Adjoint)** $P^T$:
$$
P^T = S^T \, C^T \, G^T
$$
Note that the transpose of a convolution matrix is equivalent to convolution with the "flipped kernel" (i.e., Correlation). In the frequency domain, this corresponds to multiplying by the conjugate of the PSF spectrum.
Code corresponds to `operator_solver._apply_psf_unmasked_to_unmasked(..., adjoint=True)`.
-   Uses `jnp.conj(psf_fft)` during multiplication in the frequency domain.

### 3.3 Combined Operator $A = PM$

The final operator is assembled by the `_build_forward_and_adjoint` function:

-   **Forward**: $x \to P(M(x))$
-   **Adjoint**: $y \to M^T(P^T(y))$

---

## 4. Regularization $H$ (Regularization)

The regularization matrix $H$ is used to penalize the complexity of the source. In the Operator backend, we similarly only need to implement $x \mapsto Hx$.

**Supported Modes**:
1.  **Dense GP (Gaussian Process)**: $H$ is a dense matrix.
    -   Directly computes matrix multiplication `H @ x`.
    -   Suitable for cases with fewer source pixels.
2.  **Sparse (kNN / Rectangular)**: $H$ is a sparse matrix.
    -   Stored in COO format (rows, cols, values).
    -   Computes Sparse Matrix-Vector multiplication (SpMV):
    $$
    (Hx)_i = \sum_{k} H_{i, \text{cols}_k} \, x_{\text{cols}_k}
    $$
    -   Code corresponds to `operator_solver._apply_sparse_matrix`.

Construction logic is located in [regularization.py](file:///home/cao/data_disk/tinylens_dev/TinyLensGpu/TinyLensGpu/utils/lensing/regularization.py).

---

## 5. Solving Strategy

With the function handles for $A$ and $A^T$, we can solve the inversion problem.

### 5.1 Unconstrained Solution: Conjugate Gradient (CG)

Corresponding Class: `OperatorInversion`

We solve the Normal Equations:
$$
(A^T N^{-1} A + H) s = A^T N^{-1} d
$$

To use CG, we need to define the linear operator (MatVec) of the system:
$$
\mathcal{A}(x) = A^T (N^{-1} (A x)) + H x
$$

Code implementation details:
-   `n_inv`: Pre-computed noise inverse variance vector $1/\sigma^2$.
-   `matvec(x)`:
    1.  `Ax = forward(x)`
    2.  `weighted_Ax = Ax * n_inv`
    3.  `AT_weighted_Ax = adjoint(weighted_Ax)`
    4.  `Hx = apply_H(x)`
    5.  `return AT_weighted_Ax + Hx`

**Solving with `jax.scipy.sparse.linalg.cg`:**

Although the code may use a custom `_cg_solve` for performance optimization, its core logic is consistent with the JAX standard library. Here is how to use the standard API:

-   **API Interface**:
    ```python
    x, info = jax.scipy.sparse.linalg.cg(A, b, x0=None, tol=1e-5, maxiter=None)
    ```

-   **Input Parameters**:
    -   `A`: Linear operator, corresponding to the `matvec` function defined above. In JAX, it can be a Python callable that accepts a vector `x` and returns `Ax`.
    -   `b`: The Right-Hand Side of the linear system, i.e., $A^T N^{-1} d$. It is computed by calling the adjoint operator: `b = adjoint(d * n_inv)`.
    -   `x0`: Initial guess (optional, usually set to all zeros).
    -   `tol`: Convergence tolerance (Relative Tolerance).
    -   `maxiter`: Maximum number of iterations.

-   **Output**:
    -   `x`: The solved source plane coefficient vector $s$.
    -   `info`: Convergence status info (0 indicates successful convergence).

### 5.2 Non-Negative Constrained Solution: FISTA (NNLS)

Corresponding Class: `OperatorNNLSInversion`

When $s \ge 0$ is required, the problem becomes Non-Negative Least Squares (NNLS). We use **FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)**.

Gradient of the objective function:
$$
\nabla \mathcal{L}(s) = A^T N^{-1} (As - d) + Hs
$$

**FISTA Steps**:
1.  **Gradient Descent**: $y = x_k - \eta \nabla \mathcal{L}(x_k)$
2.  **Projection (Proximal Operator)**: $x_{k+1} = \max(y, 0)$ (ReLU)
3.  **Momentum Acceleration**: Introduces a momentum term to accelerate convergence.

The step size $\eta$ depends on the Lipschitz constant (i.e., the largest eigenvalue of the Hessian). The code automatically determines the step size by estimating the largest eigenvalue via Power Iteration.

---

## 6. Log Evidence Calculation and SLQ Approximation

This is the most complex part of the Operator backend. Bayesian inference requires calculating the Log Evidence (Marginal Likelihood):

$$
\log \mathcal{Z} \approx -\mathcal{L}(s^*) - \frac{1}{2}\log|N| + \frac{1}{2}\log|H| - \frac{1}{2}\log|A^T N^{-1} A + H|
$$

The difficulty lies in the last term $\log \det (F)$, where $F = A^T N^{-1} A + H$ is a massive matrix that cannot be explicitly constructed, let alone directly eigendecomposed.

We use **SLQ (Stochastic Lanczos Quadrature)** technology to approximate the Log Determinant.

### 6.1 Mathematical Principle

Using the identity $\log \det F = \mathrm{Tr}(\log F)$.
According to the Hutchinson trace estimator:
$$
\mathrm{Tr}(\log F) \approx \frac{1}{N_p} \sum_{i=1}^{N_p} v_i^T \log(F) v_i
$$
where $v_i$ are random probe vectors (Rademacher distribution, elements are $\pm 1$).

To compute $v^T \log(F) v$ without constructing $F$, we use the **Lanczos algorithm** to approximate the projection of $F$ on the Krylov subspace generated by vector $v$ as a small tridiagonal matrix $T$:
$$
F \approx Q T Q^T
$$
Thus:
$$
v^T \log(F) v \approx \|v\|^2 \, e_1^T \log(T) e_1
$$
The size of matrix $T$ is usually only a few dozen (`slq_steps`), making it easy to compute its eigenvalues and logarithm.

### 6.2 Code Parameters

In `OperatorInversion.log_evidence`:
-   `slq_probes`: Number of random vectors $v$, determining variance.
-   `slq_steps`: Number of Lanczos iteration steps, determining approximation bias.
-   `slq_seed`: Random seed to ensure reproducibility.

This method reduces the determinant calculation complexity from $O(N^3)$ to $O(N_p \cdot N_{\text{steps}} \cdot T_{\text{matvec}})$, making Bayesian inference possible for large-scale pixelized sources (e.g., $100^2$ or higher resolution).
