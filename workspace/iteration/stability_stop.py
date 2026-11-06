"""Optional objective/field stability stopping for workspace L-BFGS experiments.

The criterion certifies small changes over a finite window, not stationarity or
global optimality. SciPy's original success/status/message remain unchanged.
"""

from collections import deque
from dataclasses import asdict, dataclass
from itertools import combinations
import time

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class StabilityPolicy:
    window: int = 50
    check_every: int = 10
    consecutive: int = 3
    objective_rtol: float = 1e-7
    solution_rtol: float = 1e-3
    objective_atol: float = 0.0
    solution_atol: float = 1e-12

    def __post_init__(self):
        counts = (self.window, self.check_every, self.consecutive)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in counts):
            raise ValueError("Window, check interval and consecutive count must be positive integers")
        if self.window % self.check_every:
            raise ValueError("Window must be a multiple of the check interval")
        tolerances = (self.objective_rtol, self.solution_rtol,
                      self.objective_atol, self.solution_atol)
        if any(not np.isfinite(v) or v < 0 for v in tolerances):
            raise ValueError("Tolerances must be finite and nonnegative")


class StabilityMonitor:
    """Check objective range and sampled field diameters in a trailing window.

Every accepted objective enters the range. Source, potential and core kappa
are sampled at check intervals and compared pairwise, so an excursion followed
by a return to the same endpoint does not masquerade as a stable window.
"""

    names = ("source", "potential", "kappa")

    def __init__(self, policy):
        self.policy = policy
        self.objectives = deque()
        self.fields = deque()
        self.records = []
        self.first_hit = None
        self.streak = 0
        self.last_iteration = -1
        self.shapes = None

    def sample_due(self, iteration):
        return iteration % self.policy.check_every == 0

    def observe(self, iteration, objective, fields=None):
        if iteration != self.last_iteration + 1:
            raise ValueError("Observe every accepted iteration, starting at zero")
        self.last_iteration = iteration
        self.objectives.append((iteration, float(objective)))
        cutoff = iteration - self.policy.window
        while self.objectives and self.objectives[0][0] < cutoff:
            self.objectives.popleft()
        if not self.sample_due(iteration):
            return False
        if fields is None:
            raise ValueError("A scheduled stability check requires physical fields")
        state = {name: np.asarray(fields[name], dtype=float).ravel().copy() for name in self.names}
        shapes = {name: value.shape for name, value in state.items()}
        if any(not value.size for value in state.values()):
            raise ValueError("Stability fields must be nonempty")
        if self.shapes is not None and shapes != self.shapes:
            raise ValueError("Stability field shapes changed during optimization")
        self.shapes = shapes
        self.fields.append((iteration, state))
        while self.fields and self.fields[0][0] < cutoff:
            self.fields.popleft()
        if iteration < self.policy.window:
            return False
        assert self.fields[0][0] == cutoff and self.objectives[0][0] == cutoff
        values = np.asarray([v for _, v in self.objectives])
        finite = bool(np.all(np.isfinite(values)) and all(
            np.all(np.isfinite(s[name])) for _, s in self.fields for name in self.names))
        record = dict(iteration=iteration, finite=finite, window_satisfied=False,
                      consecutive_satisfied=0, converged=False)
        if finite:
            span = float(np.ptp(values))
            scale = max(1.0, float(np.max(abs(values))))
            objective_ok = span <= self.policy.objective_atol + self.policy.objective_rtol * scale
            record.update(objective_span=span, objective_relative_span=span/scale,
                          objective_satisfied=bool(objective_ok))
            solution_ok = True
            states = [s for _, s in self.fields]
            for name in self.names:
                norms = (2, np.inf) if name == "kappa" else (2,)
                for norm in norms:
                    scale = max(float(np.linalg.norm(s[name], ord=norm)) for s in states)
                    diameter = max(float(np.linalg.norm(a[name]-b[name], ord=norm))
                                   for a, b in combinations(states, 2))
                    ok = diameter <= self.policy.solution_atol + self.policy.solution_rtol * scale
                    suffix = "L2" if norm == 2 else "max"
                    # Report a finite normalized change even for exactly zero fields.
                    denominator = max(scale, self.policy.solution_atol, np.finfo(float).tiny)
                    record[f"{name}_relative_{suffix}"] = diameter / denominator
                    record[f"{name}_{suffix}_satisfied"] = bool(ok)
                    solution_ok &= bool(ok)
            record["window_satisfied"] = bool(objective_ok and solution_ok)
        self.streak = self.streak + 1 if record["window_satisfied"] else 0
        record.update(consecutive_satisfied=self.streak,
                      converged=self.streak >= self.policy.consecutive)
        self.records.append(record)
        if record["converged"] and self.first_hit is None:
            self.first_hit = dict(record)
        return record["converged"]


def minimize_with_stability(fun, initial, *, initial_objective, sample_fields=None,
                            policy=None, stop_when_stable=True, options=None):
    """Run L-BFGS-B with an additional optional callback stopping condition.

fun returns (objective, gradient). sample_fields maps optimization coordinates
to separate physical source, potential and core-kappa arrays. An intentional
callback stop remains SciPy status 99; ``stability_stop`` identifies its cause.
An ordinary native exit remains available under the unchanged SciPy options.
"""
    history = []
    monitor = None if policy is None else StabilityMonitor(policy)
    monitor_seconds = 0.0
    start = time.perf_counter()
    if monitor is not None:
        if sample_fields is None:
            raise ValueError("Stability stopping requires a physical-field sampler")
        begin = time.perf_counter()
        monitor.observe(0, initial_objective, sample_fields(initial))
        monitor_seconds += time.perf_counter() - begin

    def callback(intermediate_result):
        nonlocal monitor_seconds
        history.append(float(intermediate_result.fun))
        if monitor is not None:
            iteration = len(history)
            begin = time.perf_counter()
            fields = sample_fields(intermediate_result.x) if monitor.sample_due(iteration) else None
            stable = monitor.observe(iteration, intermediate_result.fun, fields)
            monitor_seconds += time.perf_counter() - begin
            if monitor.records and monitor.records[-1]["iteration"] == iteration:
                monitor.records[-1]["elapsed_seconds"] = time.perf_counter() - start
                if monitor.first_hit is not None and monitor.first_hit["iteration"] == iteration:
                    monitor.first_hit["elapsed_seconds"] = monitor.records[-1]["elapsed_seconds"]
            if stable and stop_when_stable:
                raise StopIteration

    result = minimize(fun, initial, jac=True, method="L-BFGS-B", options=options, callback=callback)
    elapsed = time.perf_counter() - start
    stopped = bool(monitor is not None and stop_when_stable and monitor.first_hit is not None
                   and result.status == 99)
    return result, dict(objective_history=history, stability_stop=stopped,
                        optimizer_seconds=elapsed, monitor_seconds=monitor_seconds,
                        stability_policy=None if policy is None else asdict(policy),
                        first_stable_check=None if monitor is None else monitor.first_hit,
                        stability_checks=[] if monitor is None else monitor.records)
