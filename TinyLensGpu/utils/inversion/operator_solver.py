import jax
import jax.numpy as jnp
from jax import jit
from jax.tree_util import register_pytree_node_class
from functools import partial
from typing import Tuple


def _cg_solve(matvec, b: jnp.ndarray, *, tol: float, maxiter: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.zeros_like(b)
    r = b
    p = r
    rsold = jnp.dot(r, r)
    tol2 = jnp.array(float(tol) ** 2, dtype=rsold.dtype)

    def body(state, _):
        x, r, p, rsold, done = state

        Ap = matvec(p)
        denom = jnp.dot(p, Ap)
        alpha = rsold / (denom + 1e-12)

        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rsnew = jnp.dot(r_new, r_new)
        beta = rsnew / (rsold + 1e-12)
        p_new = r_new + beta * p

        done_new = done | (rsnew <= tol2)

        def _keep(_):
            return x, r, p, rsold, done

        def _update(_):
            return x_new, r_new, p_new, rsnew, done_new

        return jax.lax.cond(done, _keep, _update, operand=None), None

    (x, r, p, rsold, done), _ = jax.lax.scan(body, (x, r, p, rsold, jnp.array(False)), xs=None, length=maxiter)
    return x, rsold


def _apply_psf_unmasked_to_unmasked(
    x_unmasked: jnp.ndarray,
    psf_fft: jnp.ndarray,
    image_shape: Tuple[int, int],
    psf_shape: Tuple[int, int],
    unmasked_indices: Tuple[jnp.ndarray, jnp.ndarray],
    *,
    adjoint: bool,
) -> jnp.ndarray:
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


def _apply_mapping(
    source: jnp.ndarray,
    weights: jnp.ndarray,
    indices: jnp.ndarray,
) -> jnp.ndarray:
    vals = jnp.take(source, indices, axis=0)
    return jnp.sum(weights * vals, axis=1)


def _apply_mapping_transpose(
    x_unmasked: jnp.ndarray,
    weights: jnp.ndarray,
    indices: jnp.ndarray,
    n_source: int,
) -> jnp.ndarray:
    contrib = weights * x_unmasked[:, None]
    out = jnp.zeros((n_source,), dtype=contrib.dtype)
    out = out.at[indices.reshape(-1)].add(contrib.reshape(-1))
    return out


def _cg_and_slq_logdet(
    b: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    noise_var: jnp.ndarray,
    weights: jnp.ndarray,
    indices: jnp.ndarray,
    psf_fft: jnp.ndarray,
    image_shape: Tuple[int, int],
    psf_shape: Tuple[int, int],
    unmasked_indices: Tuple[jnp.ndarray, jnp.ndarray],
    jitter: float,
    cg_tol: float,
    cg_maxiter: int,
    slq_seed: int,
    slq_probes: int,
    slq_steps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n_data = d.shape[0]
    n_source = H.shape[0]

    N_diag = jnp.clip(noise_var, min=1e-12)
    N_inv = 1.0 / N_diag

    def A(x):
        unblur = _apply_mapping(x, weights, indices)
        return _apply_psf_unmasked_to_unmasked(
            unblur, psf_fft, image_shape, psf_shape, unmasked_indices, adjoint=False
        )

    def AT(x):
        pre = _apply_psf_unmasked_to_unmasked(
            x, psf_fft, image_shape, psf_shape, unmasked_indices, adjoint=True
        )
        return _apply_mapping_transpose(pre, weights, indices, n_source)

    def M(x):
        return AT(A(x) * N_inv) + (H @ x) + jitter * x

    s, _ = _cg_solve(M, b, tol=cg_tol, maxiter=cg_maxiter)

    d_Ninv_d = jnp.sum(d * d * N_inv)
    combined_chi2_reg = d_Ninv_d - jnp.dot(s, b)

    def lanczos_one(z):
        z = z.astype(jnp.float32)
        z_norm = jnp.linalg.norm(z)
        q = z / (z_norm + 1e-12)
        q_prev = jnp.zeros_like(q)
        beta_prev = jnp.array(0.0, dtype=q.dtype)

        def body(carry, _):
            q, q_prev, beta_prev = carry
            w = M(q) - beta_prev * q_prev
            alpha = jnp.dot(q, w)
            w = w - alpha * q
            beta = jnp.linalg.norm(w)
            q_next = w / (beta + 1e-12)
            
            # Ensure types match carry
            q_next = q_next.astype(q.dtype)
            beta = beta.astype(beta_prev.dtype)
            
            return (q_next, q, beta), (alpha, beta)

        (_, _, _), (alphas, betas) = jax.lax.scan(body, (q, q_prev, beta_prev), xs=None, length=slq_steps)
        betas = betas.at[-1].set(0.0)
        T = jnp.diag(alphas) + jnp.diag(betas[:-1], 1) + jnp.diag(betas[:-1], -1)
        evals, evecs = jnp.linalg.eigh(T)
        evals = jnp.clip(evals, min=1e-12)
        w0 = evecs[0, :] ** 2
        return (z_norm * z_norm) * jnp.sum(w0 * jnp.log(evals))

    key = jax.random.PRNGKey(slq_seed)
    zs = jax.random.rademacher(key, (slq_probes, n_source), dtype=jnp.int32)
    zs = zs.astype(jnp.float32)
    quad = jax.vmap(lanczos_one)(zs)
    logdet_M = jnp.mean(quad)
    return s, combined_chi2_reg, logdet_M


@register_pytree_node_class
class OperatorInversion:
    def __init__(
        self,
        d: jnp.ndarray,
        noise_var: jnp.ndarray,
        H: jnp.ndarray,
        weights: jnp.ndarray,
        indices: jnp.ndarray,
        psf_fft: jnp.ndarray,
        image_shape: Tuple[int, int],
        psf_shape: Tuple[int, int],
        unmasked_indices: Tuple[jnp.ndarray, jnp.ndarray],
        *,
        jitter: float = 1e-6,
        cg_tol: float = 1e-4,
        cg_maxiter: int = 40,
        slq_seed: int = 0,
        slq_probes: int = 2,
        slq_steps: int = 10,
    ) -> None:
        self.d = jnp.asarray(d, dtype=jnp.float32)
        self.noise_var = jnp.asarray(noise_var, dtype=jnp.float32)
        self.H = jnp.asarray(H, dtype=jnp.float32)
        self.weights = jnp.asarray(weights, dtype=jnp.float32)
        self.indices = jnp.asarray(indices, dtype=jnp.int32)
        self.psf_fft = psf_fft
        self.unmasked_indices = (jnp.asarray(unmasked_indices[0], dtype=jnp.int32), jnp.asarray(unmasked_indices[1], dtype=jnp.int32))

        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.psf_shape = (int(psf_shape[0]), int(psf_shape[1]))

        self.jitter = float(jitter)
        self.cg_tol = float(cg_tol)
        self.cg_maxiter = int(cg_maxiter)
        self.slq_seed = int(slq_seed)
        self.slq_probes = int(slq_probes)
        self.slq_steps = int(slq_steps)

    @jit
    def solve(self) -> jnp.ndarray:
        n_source = self.H.shape[0]
        N_diag = jnp.clip(self.noise_var, min=1e-12)
        N_inv = 1.0 / N_diag

        def A(x):
            unblur = _apply_mapping(x, self.weights, self.indices)
            return _apply_psf_unmasked_to_unmasked(
                unblur, self.psf_fft, self.image_shape, self.psf_shape, self.unmasked_indices, adjoint=False
            )

        def AT(x):
            pre = _apply_psf_unmasked_to_unmasked(
                x, self.psf_fft, self.image_shape, self.psf_shape, self.unmasked_indices, adjoint=True
            )
            return _apply_mapping_transpose(pre, self.weights, self.indices, n_source)

        def M(x):
            return AT(A(x) * N_inv) + (self.H @ x) + self.jitter * x

        b = AT(self.d * N_inv)
        s, _ = _cg_solve(M, b, tol=self.cg_tol, maxiter=self.cg_maxiter)
        return s

    @jit
    def model_predict(self, s: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the model data (unmasked pixels) given the source.
        
        Parameters
        ----------
        s : jnp.ndarray
            Source intensities, shape [n_source]
            
        Returns
        -------
        model_data : jnp.ndarray
            Model data vector, shape [n_data]
        """
        unblur = _apply_mapping(s, self.weights, self.indices)
        return _apply_psf_unmasked_to_unmasked(
            unblur, self.psf_fft, self.image_shape, self.psf_shape, self.unmasked_indices, adjoint=False
        )

    @jit
    def log_evidence(self) -> jnp.ndarray:
        n_data = self.d.shape[0]
        n_source = self.H.shape[0]

        N_diag = jnp.clip(self.noise_var, min=1e-12)
        half_log_det_N = 0.5 * jnp.sum(jnp.log(N_diag))
        log_evidence_const = -0.5 * n_data * jnp.log(2.0 * jnp.pi) - half_log_det_N

        H_stab = self.H + self.jitter * jnp.eye(n_source, dtype=self.H.dtype)
        sign_H, logdet_H = jnp.linalg.slogdet(H_stab)
        half_log_det_H = 0.5 * logdet_H

        N_inv = 1.0 / N_diag

        def AT(x):
            pre = _apply_psf_unmasked_to_unmasked(
                x, self.psf_fft, self.image_shape, self.psf_shape, self.unmasked_indices, adjoint=True
            )
            return _apply_mapping_transpose(pre, self.weights, self.indices, n_source)

        b = AT(self.d * N_inv)

        s, combined_chi2_reg, logdet_M = _cg_and_slq_logdet(
            b=b,
            H=self.H,
            d=self.d,
            noise_var=self.noise_var,
            weights=self.weights,
            indices=self.indices,
            psf_fft=self.psf_fft,
            image_shape=self.image_shape,
            psf_shape=self.psf_shape,
            unmasked_indices=self.unmasked_indices,
            jitter=self.jitter,
            cg_tol=self.cg_tol,
            cg_maxiter=self.cg_maxiter,
            slq_seed=self.slq_seed,
            slq_probes=self.slq_probes,
            slq_steps=self.slq_steps,
        )

        is_valid = (sign_H > 0) & jnp.isfinite(log_evidence_const) & jnp.isfinite(half_log_det_H) & jnp.isfinite(logdet_M)

        def _valid(_):
            log_ev = log_evidence_const
            log_ev += half_log_det_H
            log_ev -= 0.5 * combined_chi2_reg
            log_ev -= 0.5 * logdet_M
            return log_ev

        return jax.lax.cond(is_valid, _valid, lambda _: -jnp.inf, operand=None)

    def debug_terms(self):
        n_data = int(self.d.shape[0])
        n_source = int(self.H.shape[0])

        N_diag = jnp.clip(self.noise_var, min=1e-12)
        half_log_det_N = 0.5 * jnp.sum(jnp.log(N_diag))
        log_evidence_const = -0.5 * n_data * jnp.log(2.0 * jnp.pi) - half_log_det_N

        H_stab = self.H + self.jitter * jnp.eye(n_source, dtype=self.H.dtype)
        sign_H, logdet_H = jnp.linalg.slogdet(H_stab)

        N_inv = 1.0 / N_diag
        has_nan_weights = jnp.any(~jnp.isfinite(self.weights))

        def A(x):
            unblur = _apply_mapping(x, self.weights, self.indices)
            return _apply_psf_unmasked_to_unmasked(
                unblur, self.psf_fft, self.image_shape, self.psf_shape, self.unmasked_indices, adjoint=False
            )

        def AT(x):
            pre = _apply_psf_unmasked_to_unmasked(
                x, self.psf_fft, self.image_shape, self.psf_shape, self.unmasked_indices, adjoint=True
            )
            return _apply_mapping_transpose(pre, self.weights, self.indices, n_source)

        b = AT(self.d * N_inv)
        has_nan_b = jnp.any(~jnp.isfinite(b))

        def M(x):
            return AT(A(x) * N_inv) + (self.H @ x) + self.jitter * x

        test_x = jnp.ones((n_source,), dtype=jnp.float32)
        has_nan_mx = jnp.any(~jnp.isfinite(M(test_x)))

        s, res2 = _cg_solve(M, b, tol=self.cg_tol, maxiter=self.cg_maxiter)
        has_nan_s = jnp.any(~jnp.isfinite(s))

        d_Ninv_d = jnp.sum(self.d * self.d * N_inv)
        combined_chi2_reg = d_Ninv_d - jnp.dot(s, b)
        has_nan_combined = jnp.any(~jnp.isfinite(combined_chi2_reg))

        _, _, logdet_M = _cg_and_slq_logdet(
            b=b,
            H=self.H,
            d=self.d,
            noise_var=self.noise_var,
            weights=self.weights,
            indices=self.indices,
            psf_fft=self.psf_fft,
            image_shape=self.image_shape,
            psf_shape=self.psf_shape,
            unmasked_indices=self.unmasked_indices,
            jitter=self.jitter,
            cg_tol=self.cg_tol,
            cg_maxiter=self.cg_maxiter,
            slq_seed=self.slq_seed,
            slq_probes=self.slq_probes,
            slq_steps=self.slq_steps,
        )

        return {
            "cg_res2": res2,
            "sign_H": sign_H,
            "logdet_H": logdet_H,
            "log_evidence_const": log_evidence_const,
            "combined_chi2_reg": combined_chi2_reg,
            "logdet_M": logdet_M,
            "has_nan_weights": has_nan_weights,
            "has_nan_b": has_nan_b,
            "has_nan_M1": has_nan_mx,
            "has_nan_s": has_nan_s,
            "has_nan_combined": has_nan_combined,
        }

    def tree_flatten(self):
        children = (
            self.d,
            self.noise_var,
            self.H,
            self.weights,
            self.indices,
            self.psf_fft,
            self.unmasked_indices[0],
            self.unmasked_indices[1],
        )
        aux_data = (
            self.image_shape,
            self.psf_shape,
            self.jitter,
            self.cg_tol,
            self.cg_maxiter,
            self.slq_seed,
            self.slq_probes,
            self.slq_steps,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            image_shape,
            psf_shape,
            jitter,
            cg_tol,
            cg_maxiter,
            slq_seed,
            slq_probes,
            slq_steps,
        ) = aux_data
        (
            d,
            noise_var,
            H,
            weights,
            indices,
            psf_fft,
            y_indices,
            x_indices,
        ) = children
        return cls(
            d=d,
            noise_var=noise_var,
            H=H,
            weights=weights,
            indices=indices,
            psf_fft=psf_fft,
            image_shape=image_shape,
            psf_shape=psf_shape,
            unmasked_indices=(y_indices, x_indices),
            jitter=jitter,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter,
            slq_seed=slq_seed,
            slq_probes=slq_probes,
            slq_steps=slq_steps,
        )
