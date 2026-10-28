"""Independent host reference for the small nested nonlinear validation case.

This is verification code, never a production fitting backend. Coordinate and
stencil assembly comes from the independent dense-iteration reference. Every
strength pair starts at the supplied source truth and zero correction.
"""

import numpy as np
from scipy.optimize import least_squares, root

from tests.test_potential_correction_iterative import _independent_problem

STRENGTH_BOUNDS = ((0.1, 2.0), (0.001, 1.0))
SEARCH_CONTROLS = dict(maxfun=100, maxiter=100, len_tol=1e-3)


def mixed_partition():
    """Frozen cap20 incidence layout: four tiny source + sixteen potential.

    Potential columns were ranked by their initial image-response squared
    norms using an independent tiny reference. Store the calibrated indices;
    runtime fitting requires neither a full Jacobian nor topology updates.
    Neither block contains the complete potential or joint field.
    """
    return (
        (0, 1, 2, 3, 21, 20, 19, 28, 26, 27, 18, 22, 13, 12, 14, 29, 25, 11, 15, 9),
        (33, 8, 32, 31, 7, 4, 5, 6, 10, 16, 17, 23, 24, 30, 34, 35, 36),
    )


def fit_options():
    # Larger explicit iteration budget; the physical model and all stopping
    # defaults are unchanged. PCG is stricter than the true-residual gate.
    return dict(
        max_iter=240, block_size=20, block_indices=mixed_partition(), pcg_rtol=1e-12
    )


def numerical_model(problem, ls, lp):
    return _independent_problem(
        problem.renderer, problem.image_data, problem.noise_map, ls, lp, 20.0
    )


def central_jacobian(function, values, step=1e-5):
    # In this source2 case every tested query stays in the same interior cell.
    # Each image is at most quadratic in one potential coordinate; the centered
    # derivative is exact there up to floating-point evaluation error.
    return np.column_stack(
        [
            (function(values + step * unit) - function(values - step * unit))
            / (2 * step)
            for unit in np.eye(len(values))
        ]
    )


def source_cell_margin(problem, values):
    """Independent corrected-ray distance to this source2 cell's boundaries."""
    grid = problem.renderer.potential_grid
    lookup = {tuple(node): i for i, node in enumerate(np.asarray(grid.node_indices))}
    potential = values[4:]
    margins = []
    xmin, xmax, ymin, ymax = map(float, problem.renderer.source_bbox)
    for x, y in zip(
        np.asarray(problem.renderer.query_x), np.asarray(problem.renderer.query_y)
    ):
        u, v = (x - grid.origin[0]) / grid.spacing, (y - grid.origin[1]) / grid.spacing
        i, j = int(np.floor(u)), int(np.floor(v))
        a, b = u - i, v - j
        ax = ay = 0.0
        for cx, cy, weight in (
            (i, j, (1 - a) * (1 - b)),
            (i + 1, j, a * (1 - b)),
            (i, j + 1, (1 - a) * b),
            (i + 1, j + 1, a * b),
        ):
            ax += (
                weight
                * (potential[lookup[(cx + 1, cy)]] - potential[lookup[(cx - 1, cy)]])
                / (2 * grid.spacing)
            )
            ay += (
                weight
                * (potential[lookup[(cx, cy + 1)]] - potential[lookup[(cx, cy - 1)]])
                / (2 * grid.spacing)
            )
        bx, by = x - ax, y - ay
        margins.append(min(bx - xmin, xmax - bx, by - ymin, ymax - by))
    return float(min(margins))


def evaluate_reference(problem, ls, lp, values, blocks=None):
    forward, residual, prior = numerical_model(problem, ls, lp)
    image = forward(values)
    noise = np.asarray(problem.noise_active)
    jacobian = central_jacobian(forward, values)
    posterior = jacobian.T @ (jacobian / noise[:, None] ** 2) + prior
    objective = 0.5 * np.sum(residual(values) ** 2)
    fixed = 2 * objective + np.log(noise**2).sum() + len(noise) * np.log(2 * np.pi)
    full_ratio = np.linalg.slogdet(posterior)[1] - np.linalg.slogdet(prior)[1]
    block_ratio = sum(
        np.linalg.slogdet(posterior[np.ix_(block, block)])[1]
        - np.linalg.slogdet(prior[np.ix_(block, block)])[1]
        for block in (mixed_partition() if blocks is None else blocks)
    )
    gradient = (
        -jacobian.T @ ((np.asarray(problem.data_active) - image) / noise**2)
        + prior @ values
    )
    return dict(
        values=values,
        image=image,
        objective=objective,
        penalty=values @ prior @ values,
        full_log_evidence=-0.5 * (fixed + full_ratio),
        block_log_evidence=-0.5 * (fixed + block_ratio),
        scaled_gradient=np.max(np.abs(gradient) / np.sqrt(np.diag(posterior))),
        derivative_step_error=np.max(
            np.abs(jacobian - central_jacobian(forward, values, step=5e-6))
        ),
        minimum_source_cell_margin=source_cell_margin(problem, values),
    )


def independent_fit(problem, ls, lp, blocks=None):
    forward, residual, prior = numerical_model(problem, ls, lp)
    noise = np.asarray(problem.noise_active)
    data = np.asarray(problem.data_active)
    initial = np.asarray(problem.initial_values).copy()
    fitted = least_squares(
        residual,
        initial,
        jac=lambda values: central_jacobian(residual, values),
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-11,
        max_nfev=1000,
    )

    def gradient(values):
        return (
            -central_jacobian(forward, values).T @ ((data - forward(values)) / noise**2)
            + prior @ values
        )

    # Least-squares' cost gate can stop before the stationarity residual is
    # small enough for curvature comparisons. Independently polish that same
    # branch, with no state transferred from another strength candidate.
    polished = root(gradient, fitted.x, method="hybr", options={"xtol": 1e-10})
    record = evaluate_reference(problem, ls, lp, polished.x, blocks)
    record.update(
        least_squares_success=bool(fitted.success),
        least_squares_status=int(fitted.status),
        root_success=bool(polished.success),
        root_message=str(polished.message),
        nfev=int(fitted.nfev),
    )
    return record


def recovery_metrics(problem, values):
    """Raw nodal truth and centered physical derivatives on valid core only."""
    grid = problem.renderer.potential_grid
    source = np.asarray(problem.initial_values[:4])
    x, y = np.asarray(grid.x), np.asarray(grid.y)
    potential = 0.003 * np.exp(-((x - 0.07) ** 2 + (y + 0.03) ** 2) / 0.02)
    nodes = np.asarray(grid.node_indices)
    lookup = {tuple(node): i for i, node in enumerate(nodes)}

    def derivatives(field):
        dx, dy, kappa = [], [], []
        for index in np.asarray(grid.core_indices):
            i, j = nodes[index]
            left, right = field[lookup[(i - 1, j)]], field[lookup[(i + 1, j)]]
            down, up = field[lookup[(i, j - 1)]], field[lookup[(i, j + 1)]]
            dx.append((right - left) / (2 * grid.spacing))
            dy.append((up - down) / (2 * grid.spacing))
            kappa.append(
                (left + right + down + up - 4 * field[index]) / (2 * grid.spacing**2)
            )
        return np.asarray(dx), np.asarray(dy), np.asarray(kappa)

    truth = derivatives(potential)
    actual = derivatives(values[4:])
    forward, _, _ = numerical_model(problem, 1.0, 1.0)
    image_error = forward(values) - np.asarray(problem.data_active)
    rms = lambda array: float(np.sqrt(np.mean(np.asarray(array) ** 2)))
    return dict(
        source_rmse=rms(values[:4] - source),
        potential_rmse=rms(values[4:] - potential),
        deflection_rmse=rms(np.r_[actual[0] - truth[0], actual[1] - truth[1]]),
        convergence_rmse=rms(actual[2] - truth[2]),
        image_rmse=rms(image_error),
        chi2=float(np.sum((image_error / np.asarray(problem.noise_active)) ** 2)),
    )
