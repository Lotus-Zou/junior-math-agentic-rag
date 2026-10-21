"""Paired analysis for the fixed 1000-question retrieval reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


REPORT_DIR = Path(__file__).with_name("reports")
STRATEGIES = ("dense", "hybrid", "hybrid_graph")
METRICS = ("context_recall", "context_precision")


def analyze(version: str, bootstrap_samples: int, seed: int) -> dict:
    frames = {
        strategy: pd.read_csv(REPORT_DIR / f"{version}_{strategy}.csv")
        for strategy in STRATEGIES
    }
    baseline = frames["dense"]
    if len(baseline) != 1000 or baseline["question_id"].duplicated().any():
        raise ValueError("Dense baseline must contain 1000 unique question IDs")

    rng = np.random.default_rng(seed)
    comparisons = []
    for metric in METRICS:
        for strategy in STRATEGIES[1:]:
            candidate = frames[strategy]
            if len(candidate) != 1000 or candidate["question_id"].duplicated().any():
                raise ValueError(f"{strategy} must contain 1000 unique question IDs")
            paired = baseline[["question_id", metric]].merge(
                candidate[["question_id", metric]],
                on="question_id",
                how="inner",
                validate="one_to_one",
                suffixes=("_dense", f"_{strategy}"),
            )
            if len(paired) != 1000:
                raise ValueError(f"{strategy} does not match all baseline questions")

            differences = (
                paired[f"{metric}_{strategy}"] - paired[f"{metric}_dense"]
            ).to_numpy(dtype=float)
            bootstrap_means = np.empty(bootstrap_samples, dtype=float)
            for sample_index in range(bootstrap_samples):
                bootstrap_means[sample_index] = rng.choice(
                    differences, size=len(differences), replace=True
                ).mean()
            test = stats.wilcoxon(
                differences,
                zero_method="pratt",
                alternative="two-sided",
                method="auto",
            )
            standard_deviation = float(differences.std(ddof=1))
            comparisons.append(
                {
                    "metric": metric,
                    "comparison": f"{strategy} - dense",
                    "n": len(differences),
                    "mean_difference": float(differences.mean()),
                    "median_difference": float(np.median(differences)),
                    "bootstrap_95_ci_percentile": [
                        float(np.quantile(bootstrap_means, 0.025)),
                        float(np.quantile(bootstrap_means, 0.975)),
                    ],
                    "wilcoxon_statistic": float(test.statistic),
                    "p_value": float(test.pvalue),
                    "cohens_dz": (
                        float(differences.mean() / standard_deviation)
                        if standard_deviation
                        else None
                    ),
                }
            )

    _, adjusted, _, _ = multipletests(
        [item["p_value"] for item in comparisons], method="holm"
    )
    for item, adjusted_p in zip(comparisons, adjusted):
        item["p_value_holm_4_tests"] = float(adjusted_p)

    return {
        "version": version,
        "resampling_unit": "question_id, paired across retrieval strategies",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "confidence_interval": "percentile 95% CI",
        "hypothesis_test": "two-sided Wilcoxon signed-rank, zero_method=pratt",
        "multiple_comparison_correction": "Holm across 4 metric comparisons",
        "comparisons": comparisons,
        "interpretation_limit": (
            "Inference describes this fixed deterministic benchmark only; it is not "
            "a population-level estimate from real student questions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise SystemExit("--bootstrap-samples must be positive")

    result = analyze(args.version, args.bootstrap_samples, args.seed)
    output = REPORT_DIR / f"{args.version}_paired_analysis.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()