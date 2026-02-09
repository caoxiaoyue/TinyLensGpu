"""Matrix-free operator-based inversion solvers for pixelized source SLI."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit
from jax.tree_util import register_pytree_node_class


def _safe_noise_inverse(noise_var: Array, eps: float = 1e-12) -> Tuple[Array, Array]:
    """Computes safe inverse of noise variance."""
    n_diag = jnp.maximum(noise_var, eps)
    n_inv = 1.0 / n_diag
    return n_diag, n_inv


def _apply_psf_unmasked_to_unmasked(
    x_unmasked: Array,
    psf_fft: Array,
    image_shape: Tuple[int, int],
    psf_shape: Tuple[int, int],
    unmasked_indices: Tuple[Array, Array],
    *,
    adjoint: bool,
) -> Array:
    """Applies PSF convolution or correlation (adjoint) on unmasked pixels."""
    h, w = image_shape
    psf_h, psf_w = psf_shape
    y_indices, x_indices = unmasked_indices

    fft_shape = (h + psf_h - 1, w + psf_w - 1)
    start_h = (psf_h - 1) // 2
    start_w = (psf_w - 1) // 2

    if not adjoint:
        x_full = jnp.zeros((h, w), dtype=x_unmasked.dtype)
        x_full = x_full.at[y_indices, x_indices].set(x_unmasked)
        x_fft = jnp.fft.rfft2(x_full, s=fft_shape)
        y_full = jnp.fft.irfft2(x_fft * psf_fft, s=fft_shape)
        y_cropped = y_full[start_h : start_h + h, start_w : start_w + w]
        return y_cropped[y_indices, x_indices]

    y_cropped = jnp.zeros((h, w), dtype=x_unmasked.dtype)
    y_cropped = y_cropped.at[y_indices, x_indices].set(x_unmasked)
    y_padded = jnp.zeros(fft_shape, dtype=y_cropped.dtype)
    y_padded = y_padded.at[start_h : start_h + h, start_w : start_w + w].set(y_cropped)
    y_fft = jnp.fft.rfft2(y_padded, s=fft_shape)
    x_padded = jnp.fft.irfft2(y_fft * jnp.conj(psf_fft), s=fft_shape)
    x_full = x_padded[:h, :w]
    return x_full[y_indices, x_indices]


def _apply_mapping(source: Array, weights: Array, indices: Array) -> Array:
    """Maps source pixels to image plane."""
    vals = jnp.take(source, indices, axis=0)
    return jnp.sum(weights * vals, axis=1)


def _apply_mapping_transpose(x_unmasked: Array, weights: Array, indices: Array, n_source: int) -> Array:
    """Adjoint of mapping operation."""
    contrib = weights * x_unmasked[:, None]
    out = jnp.zeros((n_source,), dtype=contrib.dtype)
    out = out.at[indices.reshape(-1)].add(contrib.reshape(-1))
    return out


def _build_forward_and_adjoint(
    *,
    weights: Array,
    indices: Array,
    psf_fft: Array,
    image_shape: Tuple[int, int],
    psf_shape: Tuple[int, int],
    unmasked_indices: Tuple[Array, Array],
    n_source: int,
):
    """Builds forward and adjoint operator functions."""

    def forward(x: Array) -> Array:
        unblur = _apply_mapping(x, weights, indices)
        return _apply_psf_unmasked_to_unmasked(
            unblur,
            psf_fft,
            image_shape,
            psf_shape,
            unmasked_indices,
            adjoint=False,
        )

    def adjoint(y: Array) -> Array:
        pre = _apply_psf_unmasked_to_unmasked(
            y,
            psf_fft,
            image_shape,
            psf_shape,
            unmasked_indices,
            adjoint=True,
        )
        return _apply_mapping_transpose(pre, weights, indices, n_source)

    return forward, adjoint


def _apply_sparse_matrix(rows: Array, cols: Array, values: Array, n: int, x: Array) -> Array:
    """Applies sparse matrix multiplication."""
    y = jnp.zeros((int(n),), dtype=x.dtype)
    contrib = values * x[cols]
    y = y.at[rows].add(contrib)
    return y


def _cg_solve(matvec, b: Array, *, tol: float, maxiter: int) -> Tuple[Array, Array]:
    """Conjugate Gradient solver."""
    x = jnp.zeros_like(b)
    r = b
    p = r
    rs_old = jnp.dot(r, r)
    tol2 = jnp.array(float(tol) ** 2, dtype=rs_old.dtype)

    def body(state, _):
        x_k, r_k, p_k, rs_k, done = state

        ap = matvec(p_k)
        denom = jnp.dot(p_k, ap)
        alpha = rs_k / (denom + 1e-12)

        x_new = x_k + alpha * p_k
        r_new = r_k - alpha * ap
        rs_new = jnp.dot(r_new, r_new)
        beta = rs_new / (rs_k + 1e-12)
        p_new = r_new + beta * p_k
        done_new = done | (rs_new <= tol2)

        def keep(_):
            return x_k, r_k, p_k, rs_k, done

        def update(_):
            return x_new, r_new, p_new, rs_new, done_new

        return jax.lax.cond(done, keep, update, operand=None), None

    (x, _, _, rs_final, _), _ = jax.lax.scan(
        body,
        (x, r, p, rs_old, jnp.array(False)),
        xs=None,
        length=maxiter,
    )
    return x, rs_final


def _lanczos_logdet(matvec, n_dim: int, *, seed: int, probes: int, steps: int) -> Array:
    """Estimates log-determinant using stochastic Lanczos quadrature."""

    def lanczos_one(z: Array) -> Array:
        z = z.astype(jnp.float32)
        eps = jnp.array(1e-12, dtype=z.dtype)
        z_norm = jnp.linalg.norm(z)
        q = z / (z_norm + eps)
        q_prev = jnp.zeros_like(q)
        beta_prev = jnp.array(0.0, dtype=q.dtype)

        def body(carry, _):
            q_k, q_prev_k, beta_prev_k = carry
            w = jnp.asarray(matvec(q_k), dtype=q_k.dtype) - beta_prev_k * q_prev_k
            alpha = jnp.asarray(jnp.dot(q_k, w), dtype=q_k.dtype)
            w = w - alpha * q_k
            beta = jnp.asarray(jnp.linalg.norm(w), dtype=q_k.dtype)
            q_next = w / (beta + eps)
            return (q_next, q_k, beta), (alpha, beta)

        (_, _, _), (alphas, betas) = jax.lax.scan(body, (q, q_prev, beta_prev), xs=None, length=steps)
        betas = betas.at[-1].set(jnp.array(0.0, dtype=betas.dtype))
        t = jnp.diag(alphas) + jnp.diag(betas[:-1], 1) + jnp.diag(betas[:-1], -1)
        eigvals, eigvecs = jnp.linalg.eigh(t)
        eigvals = jnp.maximum(eigvals, jnp.array(1e-12, dtype=eigvals.dtype))
        w0 = eigvecs[0, :] ** 2
        return (z_norm * z_norm) * jnp.sum(w0 * jnp.log(eigvals))

    key = jax.random.PRNGKey(int(seed))
    z = jax.random.rademacher(key, (int(probes), int(n_dim)), dtype=jnp.int32).astype(jnp.float32)
    values = jax.vmap(lanczos_one)(z)
    return jnp.mean(values)


def _choose_slq_size(evidence_mode: str, probes: int, steps: int) -> Tuple[int, int]:
    """Selects SLQ probes and steps based on evidence mode."""
    if evidence_mode == "fast":
        return max(4, min(int(probes), 8)), max(10, min(int(steps), 20))
    return int(probes), int(steps)


def _estimate_lipschitz_power_iteration(
    grad_fn,
    n_dim: int,
    *,
    n_iter: int = 12,
    seed: int = 0,
) -> Array:
    """Estimates Lipschitz constant via power iteration."""
    key = jax.random.PRNGKey(seed)
    v = jax.random.normal(key, (n_dim,), dtype=jnp.float32)
    eps = jnp.array(1e-12, dtype=v.dtype)
    v = v / (jnp.linalg.norm(v) + eps)

    def body(vec, _):
        w = jnp.asarray(grad_fn(vec), dtype=vec.dtype)
        nrm = jnp.asarray(jnp.linalg.norm(w), dtype=vec.dtype)
        return w / (nrm + eps), nrm

    _, norms = jax.lax.scan(body, v, xs=None, length=n_iter)
    return jnp.maximum(norms[-1], jnp.array(1e-6, dtype=norms.dtype))


class _OperatorSolverBase:
    """Base class for operator-based solvers containing common logic."""

    def __init__(
        self,
        d: Array,
        noise_var: Array,
        H: Array,
        weights: Array,
        indices: Array,
        psf_fft: Array,
        image_shape: Tuple[int, int],
        psf_shape: Tuple[int, int],
        unmasked_indices: Tuple[Array, Array],
        *,
        jitter: float,
        slq_seed: int,
        slq_probes: int,
        slq_steps: int,
        dense_logdet_max_n: int,
        evidence_mode: str,
        reg_operator_mode: str,
        H_sparse_rows: Array | None,
        H_sparse_cols: Array | None,
        H_sparse_values: Array | None,
        H_sparse_n_source: int | None,
    ) -> None:
        self.d = jnp.asarray(d, dtype=jnp.float32)
        self.noise_var = jnp.asarray(noise_var, dtype=jnp.float32)
        self.H = jnp.asarray(H, dtype=jnp.float32)
        self.weights = jnp.asarray(weights, dtype=jnp.float32)
        self.indices = jnp.asarray(indices, dtype=jnp.int32)
        self.psf_fft = jnp.asarray(psf_fft, dtype=jnp.complex64)
        self.unmasked_indices = (
            jnp.asarray(unmasked_indices[0], dtype=jnp.int32),
            jnp.asarray(unmasked_indices[1], dtype=jnp.int32),
        )

        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.psf_shape = (int(psf_shape[0]), int(psf_shape[1]))

        self.jitter = float(jitter)
        self.slq_seed = int(slq_seed)
        self.slq_probes = int(slq_probes)
        self.slq_steps = int(slq_steps)
        self.dense_logdet_max_n = int(dense_logdet_max_n)
        self.evidence_mode = str(evidence_mode).strip().lower()
        self.reg_operator_mode = str(reg_operator_mode).strip().lower()

        if self.reg_operator_mode not in {"dense_gp", "sparse_knn"}:
            raise ValueError(
                f"Unknown reg_operator_mode: '{reg_operator_mode}'. Must be one of {'dense_gp', 'sparse_knn'}."
            )

        if H_sparse_rows is None:
            H_sparse_rows = jnp.zeros((0,), dtype=jnp.int32)
        if H_sparse_cols is None:
            H_sparse_cols = jnp.zeros((0,), dtype=jnp.int32)
        if H_sparse_values is None:
            H_sparse_values = jnp.zeros((0,), dtype=jnp.float32)
        self.H_sparse_rows = jnp.asarray(H_sparse_rows, dtype=jnp.int32)
        self.H_sparse_cols = jnp.asarray(H_sparse_cols, dtype=jnp.int32)
        self.H_sparse_values = jnp.asarray(H_sparse_values, dtype=jnp.float32)
        self.H_sparse_n_source = int(H_sparse_n_source) if H_sparse_n_source is not None else int(self.H.shape[0])

        if self.reg_operator_mode == "sparse_knn":
            if self.H_sparse_values.shape[0] == 0:
                raise ValueError(
                    "reg_operator_mode='sparse_knn' requires non-empty sparse regularization entries."
                )
            self.n_source = int(self.H_sparse_n_source)
        else:
            if self.H.ndim != 2 or self.H.shape[0] != self.H.shape[1]:
                raise ValueError("Dense regularization mode requires square dense matrix H.")
            self.n_source = int(self.H.shape[0])

    def _ops(self):
        return _build_forward_and_adjoint(
            weights=self.weights,
            indices=self.indices,
            psf_fft=self.psf_fft,
            image_shape=self.image_shape,
            psf_shape=self.psf_shape,
            unmasked_indices=self.unmasked_indices,
            n_source=self.n_source,
        )

    def _apply_H(self, x: Array) -> Array:
        if self.reg_operator_mode != "sparse_knn" or self.H_sparse_values.shape[0] == 0:
            return self.H @ x
        return _apply_sparse_matrix(self.H_sparse_rows, self.H_sparse_cols, self.H_sparse_values, self.H_sparse_n_source, x)

    def _half_log_det_H(self) -> Tuple[Array, Array]:
        n_source = self.n_source
        if self.reg_operator_mode != "sparse_knn" or self.H_sparse_values.shape[0] == 0:
            h_stab = self.H + self.jitter * jnp.eye(n_source, dtype=self.H.dtype)
            sign_h, logdet_h = jnp.linalg.slogdet(h_stab)
            return sign_h, 0.5 * logdet_h

        def hvec(v: Array) -> Array:
            return self._apply_H(v) + self.jitter * v

        if n_source <= self.dense_logdet_max_n:
            eye = jnp.eye(n_source, dtype=self.d.dtype)
            h_dense = jax.vmap(hvec, in_axes=1, out_axes=1)(eye)
            sign_h, logdet_h = jnp.linalg.slogdet(h_dense)
            return sign_h, 0.5 * logdet_h

        probes, steps = _choose_slq_size(self.evidence_mode, self.slq_probes, self.slq_steps)
        logdet_h = _lanczos_logdet(hvec, n_source, seed=self.slq_seed + 113, probes=probes, steps=steps)
        return jnp.array(1.0, dtype=self.d.dtype), 0.5 * logdet_h

    @jit
    def model_predict(self, x: Array) -> Array:
        forward, _ = self._ops()
        return forward(jnp.asarray(x, dtype=jnp.float32))

    @jit
    def objective_value(self, x: Array) -> Array:
        _, n_inv = _safe_noise_inverse(self.noise_var)
        model = self.model_predict(x)
        resid = self.d - model
        chi2 = jnp.sum((resid * resid) * n_inv)
        reg = jnp.dot(x, self._apply_H(x)) + self.jitter * jnp.dot(x, x)
        return chi2 + reg

    def _get_common_children_aux(self):
        children = (
            self.d,
            self.noise_var,
            self.H,
            self.weights,
            self.indices,
            self.psf_fft,
            self.unmasked_indices[0],
            self.unmasked_indices[1],
            self.H_sparse_rows,
            self.H_sparse_cols,
            self.H_sparse_values,
        )
        aux_data = (
            self.image_shape,
            self.psf_shape,
            self.jitter,
            self.slq_seed,
            self.slq_probes,
            self.slq_steps,
            self.dense_logdet_max_n,
            self.evidence_mode,
            self.reg_operator_mode,
            self.H_sparse_n_source,
        )
        return children, aux_data


@register_pytree_node_class
class OperatorInversion(_OperatorSolverBase):
    """Operator-based semi-linear inversion using CG + SLQ."""

    def __init__(
        self,
        d: Array,
        noise_var: Array,
        H: Array,
        weights: Array,
        indices: Array,
        psf_fft: Array,
        image_shape: Tuple[int, int],
        psf_shape: Tuple[int, int],
        unmasked_indices: Tuple[Array, Array],
        *,
        jitter: float = 1e-6,
        cg_tol: float = 1e-6,
        cg_maxiter: int = 300,
        slq_seed: int = 0,
        slq_probes: int = 32,
        slq_steps: int = 60,
        dense_logdet_max_n: int = 256,
        evidence_mode: str = "accurate",
        reg_operator_mode: str = "dense_gp",
        H_sparse_rows: Array | None = None,
        H_sparse_cols: Array | None = None,
        H_sparse_values: Array | None = None,
        H_sparse_n_source: int | None = None,
    ) -> None:
        super().__init__(
            d, noise_var, H, weights, indices, psf_fft, image_shape, psf_shape, unmasked_indices,
            jitter=jitter, slq_seed=slq_seed, slq_probes=slq_probes, slq_steps=slq_steps,
            dense_logdet_max_n=dense_logdet_max_n, evidence_mode=evidence_mode,
            reg_operator_mode=reg_operator_mode, H_sparse_rows=H_sparse_rows,
            H_sparse_cols=H_sparse_cols, H_sparse_values=H_sparse_values,
            H_sparse_n_source=H_sparse_n_source
        )
        self.cg_tol = float(cg_tol)
        self.cg_maxiter = int(cg_maxiter)

    @jit
    def solve(self) -> Array:
        n_source = self.n_source
        _, n_inv = _safe_noise_inverse(self.noise_var)
        forward, adjoint = self._ops()

        def mvec(x: Array) -> Array:
            return adjoint(forward(x) * n_inv) + self._apply_H(x) + self.jitter * x

        b = adjoint(self.d * n_inv)
        x, _ = _cg_solve(mvec, b, tol=self.cg_tol, maxiter=self.cg_maxiter)
        return x[:n_source]

    @jit
    def log_evidence(self) -> Array:
        n_data = self.d.shape[0]
        n_source = self.n_source
        n_diag, n_inv = _safe_noise_inverse(self.noise_var)

        half_log_det_n = 0.5 * jnp.sum(jnp.log(n_diag))
        log_evidence_const = -0.5 * n_data * jnp.log(2.0 * jnp.pi) - half_log_det_n

        sign_h, half_log_det_h = self._half_log_det_H()

        forward, adjoint = self._ops()

        def mvec(x: Array) -> Array:
            return adjoint(forward(x) * n_inv) + self._apply_H(x) + self.jitter * x

        b = adjoint(self.d * n_inv)
        s, _ = _cg_solve(mvec, b, tol=self.cg_tol, maxiter=self.cg_maxiter)

        d_ninv_d = jnp.sum(self.d * self.d * n_inv)
        combined_chi2_reg = d_ninv_d - jnp.dot(s, b)
        n_source_int = int(n_source)
        probes, steps = _choose_slq_size(self.evidence_mode, self.slq_probes, self.slq_steps)
        if self.evidence_mode != "fast" and n_source_int <= self.dense_logdet_max_n:
            eye = jnp.eye(n_source_int, dtype=self.H.dtype)
            m_dense = jax.vmap(mvec, in_axes=1, out_axes=1)(eye)
            _, logdet_m = jnp.linalg.slogdet(m_dense)
        else:
            logdet_m = _lanczos_logdet(
                mvec,
                n_source_int,
                seed=self.slq_seed,
                probes=probes,
                steps=steps,
            )

        is_valid = (
            (sign_h > 0)
            & jnp.isfinite(log_evidence_const)
            & jnp.isfinite(half_log_det_h)
            & jnp.isfinite(logdet_m)
        )

        def valid(_):
            val = log_evidence_const
            val += half_log_det_h
            val -= 0.5 * combined_chi2_reg
            val -= 0.5 * logdet_m
            return val

        return jax.lax.cond(is_valid, valid, lambda _: -jnp.inf, operand=None)

    def tree_flatten(self):
        children, aux = self._get_common_children_aux()
        aux_final = aux + (self.cg_tol, self.cg_maxiter)
        return children, aux_final

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            image_shape, psf_shape, jitter, slq_seed, slq_probes, slq_steps,
            dense_logdet_max_n, evidence_mode, reg_operator_mode, H_sparse_n_source,
            cg_tol, cg_maxiter
        ) = aux_data
        (
            d, noise_var, H, weights, indices, psf_fft, y_indices, x_indices,
            H_sparse_rows, H_sparse_cols, H_sparse_values
        ) = children
        return cls(
            d=d, noise_var=noise_var, H=H, weights=weights, indices=indices,
            psf_fft=psf_fft, image_shape=image_shape, psf_shape=psf_shape,
            unmasked_indices=(y_indices, x_indices), jitter=jitter,
            cg_tol=cg_tol, cg_maxiter=cg_maxiter,
            slq_seed=slq_seed, slq_probes=slq_probes, slq_steps=slq_steps,
            dense_logdet_max_n=dense_logdet_max_n, evidence_mode=evidence_mode,
            reg_operator_mode=reg_operator_mode, H_sparse_rows=H_sparse_rows,
            H_sparse_cols=H_sparse_cols, H_sparse_values=H_sparse_values,
            H_sparse_n_source=H_sparse_n_source,
        )


@register_pytree_node_class
class OperatorNNLSInversion(_OperatorSolverBase):
    """Operator-based nonnegative inversion using FISTA projection."""

    def __init__(
        self,
        d: Array,
        noise_var: Array,
        H: Array,
        weights: Array,
        indices: Array,
        psf_fft: Array,
        image_shape: Tuple[int, int],
        psf_shape: Tuple[int, int],
        unmasked_indices: Tuple[Array, Array],
        *,
        jitter: float = 1e-6,
        maxiter: int = 600,
        tol: float = 1e-6,
        lipschitz_iters: int = 12,
        fista_seed: int = 0,
        evidence_mode: str = "accurate",
        slq_seed: int = 0,
        slq_probes: int = 32,
        slq_steps: int = 60,
        dense_logdet_max_n: int = 256,
        reg_operator_mode: str = "dense_gp",
        H_sparse_rows: Array | None = None,
        H_sparse_cols: Array | None = None,
        H_sparse_values: Array | None = None,
        H_sparse_n_source: int | None = None,
    ) -> None:
        super().__init__(
            d, noise_var, H, weights, indices, psf_fft, image_shape, psf_shape, unmasked_indices,
            jitter=jitter, slq_seed=slq_seed, slq_probes=slq_probes, slq_steps=slq_steps,
            dense_logdet_max_n=dense_logdet_max_n, evidence_mode=evidence_mode,
            reg_operator_mode=reg_operator_mode, H_sparse_rows=H_sparse_rows,
            H_sparse_cols=H_sparse_cols, H_sparse_values=H_sparse_values,
            H_sparse_n_source=H_sparse_n_source
        )
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.lipschitz_iters = int(lipschitz_iters)
        self.fista_seed = int(fista_seed)

    @jit
    def _gradient(self, x: Array) -> Array:
        _, n_inv = _safe_noise_inverse(self.noise_var)
        forward, adjoint = self._ops()
        resid = forward(x) - self.d
        return adjoint(resid * n_inv) + self._apply_H(x) + self.jitter * x

    @jit
    def solve(self) -> Array:
        n_source = self.n_source
        grad_fn = lambda vec: self._gradient(vec)

        l_est = _estimate_lipschitz_power_iteration(
            grad_fn,
            n_source,
            n_iter=self.lipschitz_iters,
            seed=self.fista_seed,
        )
        step = 1.0 / (l_est + 1e-12)

        x0 = jnp.zeros((n_source,), dtype=jnp.float32)
        y0 = x0
        t0 = jnp.array(1.0, dtype=jnp.float32)
        obj0 = self.objective_value(x0)

        def body(state, _):
            x_prev, y_prev, t_prev, obj_prev, done = state

            grad_y = grad_fn(y_prev)
            x_next = jnp.maximum(y_prev - step * grad_y, 0.0)
            t_next = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t_prev * t_prev))
            momentum = (t_prev - 1.0) / (t_next + 1e-12)
            y_next = x_next + momentum * (x_next - x_prev)

            obj_next = self.objective_value(x_next)
            rel = jnp.abs(obj_prev - obj_next) / (jnp.abs(obj_prev) + 1e-12)
            done_next = done | (rel <= self.tol)

            def keep(_):
                return x_prev, y_prev, t_prev, obj_prev, done

            def update(_):
                return x_next, y_next, t_next, obj_next, done_next

            return jax.lax.cond(done, keep, update, operand=None), None

        (x_final, _, _, _, _), _ = jax.lax.scan(
            body,
            (x0, y0, t0, obj0, jnp.array(False)),
            xs=None,
            length=self.maxiter,
        )
        return x_final

    @jit
    def log_evidence(self) -> Array:
        n_data = self.d.shape[0]
        n_source = self.n_source
        n_diag, n_inv = _safe_noise_inverse(self.noise_var)

        half_log_det_n = 0.5 * jnp.sum(jnp.log(n_diag))
        log_evidence_const = -0.5 * n_data * jnp.log(2.0 * jnp.pi) - half_log_det_n

        sign_h, half_log_det_h = self._half_log_det_H()

        x = self.solve()
        obj = self.objective_value(x)

        forward, adjoint = self._ops()

        def mvec(v: Array) -> Array:
            return adjoint(forward(v) * n_inv) + self._apply_H(v) + self.jitter * v

        n_source_int = int(n_source)
        probes, steps = _choose_slq_size(self.evidence_mode, self.slq_probes, self.slq_steps)
        if self.evidence_mode != "fast" and n_source_int <= self.dense_logdet_max_n:
            eye = jnp.eye(n_source_int, dtype=self.H.dtype)
            m_dense = jax.vmap(mvec, in_axes=1, out_axes=1)(eye)
            _, logdet_m = jnp.linalg.slogdet(m_dense)
        else:
            logdet_m = _lanczos_logdet(
                mvec,
                n_source_int,
                seed=self.slq_seed,
                probes=probes,
                steps=steps,
            )

        is_valid = (sign_h > 0) & jnp.isfinite(log_evidence_const) & jnp.isfinite(half_log_det_h) & jnp.isfinite(logdet_m)

        def valid(_):
            val = log_evidence_const
            val += half_log_det_h
            val -= 0.5 * obj
            val -= 0.5 * logdet_m
            return val

        return jax.lax.cond(is_valid, valid, lambda _: -jnp.inf, operand=None)

    def tree_flatten(self):
        children, aux = self._get_common_children_aux()
        aux_final = aux + (self.maxiter, self.tol, self.lipschitz_iters, self.fista_seed)
        return children, aux_final

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            image_shape, psf_shape, jitter, slq_seed, slq_probes, slq_steps,
            dense_logdet_max_n, evidence_mode, reg_operator_mode, H_sparse_n_source,
            maxiter, tol, lipschitz_iters, fista_seed
        ) = aux_data
        (
            d, noise_var, H, weights, indices, psf_fft, y_indices, x_indices,
            H_sparse_rows, H_sparse_cols, H_sparse_values
        ) = children
        return cls(
            d=d, noise_var=noise_var, H=H, weights=weights, indices=indices,
            psf_fft=psf_fft, image_shape=image_shape, psf_shape=psf_shape,
            unmasked_indices=(y_indices, x_indices), jitter=jitter,
            maxiter=maxiter, tol=tol, lipschitz_iters=lipschitz_iters, fista_seed=fista_seed,
            slq_seed=slq_seed, slq_probes=slq_probes, slq_steps=slq_steps,
            dense_logdet_max_n=dense_logdet_max_n, evidence_mode=evidence_mode,
            reg_operator_mode=reg_operator_mode, H_sparse_rows=H_sparse_rows,
            H_sparse_cols=H_sparse_cols, H_sparse_values=H_sparse_values,
            H_sparse_n_source=H_sparse_n_source,
        )


__all__ = [
    "OperatorInversion",
    "OperatorNNLSInversion",
    "_apply_psf_unmasked_to_unmasked",
    "_apply_mapping",
    "_apply_mapping_transpose",
]
