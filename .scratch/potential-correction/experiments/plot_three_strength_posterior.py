"""Weighted Nautilus corner plot and degeneracy diagnostics, without fitting.

This is an exploratory posterior diagnostic. Summaries use all importance
weights, and displayed axes contain central 99.9% marginal probability.
Contours enclose 68% and 95% joint probability. No Gaussian approximation
is used for the plotted posterior or reported marginal intervals.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import corner


def quantile(values, weights, probabilities):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cumulative = (np.cumsum(w)-.5*w) / w.sum()
    return np.interp(probabilities, cumulative, v)


def moments(points, weights):
    weights = weights/weights.sum()
    mean = np.sum(points*weights[:, None], axis=0)
    delta = points-mean
    covariance = np.einsum("ni,n,nj->ij", delta, weights, delta)
    sd = np.sqrt(np.diag(covariance))
    correlation = covariance/(sd[:, None]*sd[None, :])
    return mean, covariance, correlation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run/"manifest.json").read_text())
    completion = json.loads((args.run/"completion.json").read_text())
    assert completion["complete"], "Do not label an unfinished sampler as a completed posterior"
    with np.load(args.run/"posterior.npz", allow_pickle=False) as archive:
        points, weights = archive["log10_strengths"], archive["weights"]
        log_likelihood = archive["conditional_log_evidence"]
        names = archive["parameter_order"].tolist()
    weights = weights/weights.sum()
    mean, covariance, correlation = moments(points, weights)
    ranks = np.empty_like(points)
    for i in range(3):
        order = np.argsort(points[:, i])
        ranks[order, i] = np.cumsum(weights[order])-.5*weights[order]
    spearman = moments(ranks, weights)[2]
    precision = np.linalg.inv(covariance)
    partial = -precision / np.sqrt(np.diag(precision)[:, None]*np.diag(precision)[None, :])
    np.fill_diagonal(partial, 1.)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    bounds = np.asarray(manifest["sampler_settings"]["log10_prior_bounds"])
    probabilities = np.array([.025, .16, .5, .84, .975])
    summaries = {}
    for i, name in enumerate(names):
        q = quantile(points[:, i], weights, probabilities)
        summaries[name] = dict(
            log10_quantiles=q.tolist(), linear_quantiles=(10.0**q).tolist(),
            probabilities=probabilities.tolist(), log10_mean=float(mean[i]),
            log10_standard_deviation=float(np.sqrt(covariance[i, i])),
            estimated_mass_within_0_1_dex_of_lower_bound=float(weights[points[:, i] < bounds[i, 0]+.1].sum()),
            estimated_mass_within_0_1_dex_of_upper_bound=float(weights[points[:, i] > bounds[i, 1]-.1].sum()),
        )
    split = []
    for parity in (0, 1):
        indices = np.arange(len(points)) % 2 == parity
        w = weights[indices]/weights[indices].sum()
        split.append(dict(parity=parity, kish_effective_size=float(1/(w@w)),
                          pearson_log10=moments(points[indices], w)[2].tolist()))
    report = dict(
        parameter_order=names, hyperpriors=manifest["sampler_settings"]["log10_prior_bounds"],
        sampler_settings=manifest["sampler_settings"], completion=completion,
        posterior_points=len(points), kish_effective_sample_size=float(1/(weights@weights)),
        quantiles=summaries, pearson_log10=correlation.tolist(),
        pearson_linear=moments(10.0**points, weights)[2].tolist(),
        weighted_spearman=spearman.tolist(), partial_correlation_log10=partial.tolist(),
        log10_covariance=covariance.tolist(), standardized_correlation_eigenvalues=eigenvalues.tolist(),
        standardized_correlation_eigenvectors=eigenvectors.tolist(),
        standardized_correlation_condition_number=float(eigenvalues.max()/eigenvalues.min()),
        alternating_sample_diagnostic=split,
        maximum_sampled_conditional_log_evidence=float(log_likelihood.max()),
        plot=dict(coordinates="log10 strengths", display_order=[names[i] for i in [2, 1, 0]],
                  displayed_marginal_probability=.999, joint_contours=[.68, .95],
                  weights="All normalized importance weights; no equal-weight resampling",
                  smoothing="0.8 histogram bins; summary statistics are unsmoothed"),
        caution="Conditional on fixed true macro and frozen truth gradient. A single n_live=300 run and alternating-index checks do not establish between-run convergence.",
        posterior_sha256=hashlib.sha256((args.run/"posterior.npz").read_bytes()).hexdigest(),
    )
    (args.run/"posterior-summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    order = [2, 1, 0]
    labels = [r"$\log_{10}\lambda_0$", r"$\log_{10}\lambda_4$", r"$\log_{10}\lambda_s$"]
    reference = Path(__file__).resolve().parents[1]/"research/three-strength-single-step"/manifest["sampler_settings"]["case"]/"summary.json"
    optimum = np.asarray(json.loads(reference.read_text())["selected"]["log10_strengths"])
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = corner.corner(
        points[:, order], weights=weights, labels=labels, range=[.999]*3,
        bins=45, smooth=.8, smooth1d=.8, quantiles=[.16, .5, .84],
        show_titles=True, title_fmt=".3f", color="#277b8b",
        truths=optimum[order], truth_color="#ba6a31", levels=[.68, .95],
        plot_datapoints=False, fill_contours=True,
        max_n_ticks=4, label_kwargs={"fontsize": 12},
        title_kwargs={"fontsize": 10}, fig=plt.figure(figsize=(8.5, 8.5)),
    )
    axes = np.asarray(fig.axes).reshape(3, 3)
    for i in range(3):
        for j in range(i):
            axes[i, j].text(.05, .95, rf"$\rho={correlation[order[i], order[j]]:+.3f}$",
                            transform=axes[i, j].transAxes, va="top", fontsize=10)
    fig.subplots_adjust(top=.88, left=.12, bottom=.10, right=.97)
    fig.suptitle(f"Nautilus hyperposterior: {manifest['sampler_settings']['n_live']} live points", y=.98, fontsize=14)
    fig.text(.5, .94, f"Weighted effective sample size: {report['kish_effective_sample_size']:.0f}; 68% / 95% joint contours",
             ha="center", fontsize=10)
    fig.text(.5, .913, "Orange lines: previous evidence optimum; axes show central 99.9% marginal probability",
             ha="center", fontsize=9)
    fig.savefig(args.run/"posterior_corner.png", dpi=200)
    fig.savefig(args.run/"posterior_corner.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.8), layout="constrained")
    image = ax.imshow(correlation[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(3):
        for j in range(3):
            value = correlation[order[i], order[j]]
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center",
                    color="white" if abs(value)>.6 else "black", fontsize=12)
    ax.set_xticks(range(3), labels=labels)
    ax.set_yticks(range(3), labels=labels)
    ax.set_title("Weighted posterior correlation (log strengths)", fontsize=11)
    fig.colorbar(image, ax=ax, label=r"Pearson $\rho$", shrink=.85)
    fig.savefig(args.run/"posterior_correlation.png", dpi=200)
    fig.savefig(args.run/"posterior_correlation.pdf")
    plt.close(fig)
    print(json.dumps(dict(quantiles=summaries, parameter_order=names,
                          pearson_log10=correlation.tolist(), weighted_spearman=spearman.tolist(),
                          correlation_condition_number=report["standardized_correlation_condition_number"])), flush=True)


if __name__ == "__main__":
    main()
