"""Reproduce the protocol's planning power estimates using synthetic outcomes.

Run with Python 3 and NumPy 2.4.6: python power_simulation.py
No patient data or rating responses are read. Each scenario uses 10,000 simulated
studies and the specified 1,000-replicate case-level percentile bootstrap.
"""

import json

import numpy as np


SEED = 20260925
N_STUDIES = 10_000
N_BOOTSTRAPS = 1_000
N_COMMON = 15
N_UNIQUE_PER_RATER = 75
BATCH_SIZE = 100


def bootstrap_intervals(category_counts, successes, sizes, rng):
    """Resample cases, retaining every judgement of each selected case.

    Cases with identical numbers of correct and total judgements are grouped.
    Multinomial sampling of these groups is distributionally identical to
    sampling individual cases uniformly with replacement.
    """
    n_cases = int(category_counts[0].sum())
    draws = rng.multinomial(
        n_cases,
        category_counts / n_cases,
        size=(N_BOOTSTRAPS, len(category_counts)),
    )
    accuracy = (draws @ successes) / (draws @ sizes)
    return np.quantile(accuracy, [0.025, 0.975], axis=0, method="linear")


def simulate(n_raters, probability, dependence):
    """Estimate decision probabilities for one complete-response scenario."""
    scenario = {"independent": 0, "concordant": 1}[dependence]
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [SEED, n_raters, int(round(100 * probability)), scenario]
        )
    )
    n_unique = N_UNIQUE_PER_RATER * n_raters
    # Categories: incorrect/correct unique cases, then common cases with 0..R
    # correct judgements. Every common case contributes R judgements.
    successes = np.r_[0, 1, np.arange(n_raters + 1)]
    sizes = np.r_[1, 1, np.full(n_raters + 1, n_raters)]
    counts = np.zeros(3, dtype=np.int64)

    for start in range(0, N_STUDIES, BATCH_SIZE):
        batch_size = min(BATCH_SIZE, N_STUDIES - start)
        unique_correct = rng.binomial(n_unique, probability, size=batch_size)
        if dependence == "independent":
            common_correct = rng.binomial(
                n_raters, probability, size=(batch_size, N_COMMON)
            )
        else:
            common_correct = n_raters * rng.binomial(
                1, probability, size=(batch_size, N_COMMON)
            )

        categories = np.zeros((batch_size, n_raters + 3), dtype=np.int64)
        categories[:, 0] = n_unique - unique_correct
        categories[:, 1] = unique_correct
        for k in range(n_raters + 1):
            categories[:, k + 2] = (common_correct == k).sum(axis=1)

        lower, upper = bootstrap_intervals(categories, successes, sizes, rng)
        counts += [
            np.count_nonzero((lower > 0.5) | (upper < 0.5)),
            np.count_nonzero((lower >= 0.40) & (upper <= 0.60)),
            np.count_nonzero((lower >= 0.425) & (upper <= 0.575)),
        ]

    powers = counts / N_STUDIES
    return {
        "n_raters": n_raters,
        "n_cases": N_COMMON + n_unique,
        "accuracy": probability,
        "common_case_dependence": dependence,
        "excludes_0.5": float(powers[0]),
        "within_0.40_0.60": float(powers[1]),
        "within_0.425_0.575": float(powers[2]),
        "monte_carlo_se": np.sqrt(
            powers * (1 - powers) / N_STUDIES
        ).tolist(),
    }


def main():
    print(json.dumps({
        "numpy_version": np.__version__,
        "seed": SEED,
        "studies_per_scenario": N_STUDIES,
        "bootstrap_replicates": N_BOOTSTRAPS,
    }), flush=True)
    for n_raters in (6, 5, 4, 3, 2):
        for dependence in ("independent", "concordant"):
            for probability in (0.6, 0.5):
                print(json.dumps(simulate(n_raters, probability, dependence)),
                      flush=True)


if __name__ == "__main__":
    main()
