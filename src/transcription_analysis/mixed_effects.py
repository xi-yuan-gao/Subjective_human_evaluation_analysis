from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from pathlib import Path
from typing import Any

from .config import load_config
from .io_utils import read_csv, write_csv


REQUIRED_COLUMNS = {
    "participant_id",
    "session_id",
    "utterance_id",
    "condition",
    "speaker_id",
    "wer",
    "cer",
}


def _load_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import pandas as pd
        import scipy
        import statsmodels
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise RuntimeError(
            "Mixed-effects dependencies are not installed. Install the optional "
            "extra with: python -m pip install -e '.[mixed-effects]'"
        ) from exc
    return np, pd, scipy, (statsmodels, smf)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_data(
    scores_path: Path,
    expected_conditions: list[str],
) -> Any:
    _, pd, _, _ = _load_dependencies()
    rows = read_csv(scores_path)
    if not rows:
        raise ValueError(f"Scores file is empty: {scores_path}")
    missing_columns = sorted(REQUIRED_COLUMNS - set(rows[0]))
    if missing_columns:
        raise ValueError(
            "Scores file is missing required columns: " + ", ".join(missing_columns)
        )

    data = pd.DataFrame(rows)
    for column in ("participant_id", "session_id", "utterance_id", "condition", "speaker_id"):
        data[column] = data[column].astype(str).str.strip()
        if (data[column] == "").any():
            raise ValueError(f"Scores file contains blank {column} values.")
    for outcome in ("wer", "cer"):
        data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
        if data[outcome].isna().any():
            raise ValueError(f"Scores file contains missing or non-numeric {outcome} values.")
        if (data[outcome] < 0).any():
            raise ValueError(f"Scores file contains negative {outcome} values.")

    observed_conditions = set(data["condition"])
    expected_set = set(expected_conditions)
    if observed_conditions != expected_set:
        missing = sorted(expected_set - observed_conditions)
        unexpected = sorted(observed_conditions - expected_set)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ValueError("Condition levels do not match the configuration (" + "; ".join(details) + ").")

    data["condition"] = pd.Categorical(
        data["condition"], categories=expected_conditions, ordered=True
    )
    data["listener_id"] = data["participant_id"] + "::" + data["session_id"]
    data["all_rows"] = "all"
    return data


def _condition_contrast(
    result: Any,
    condition_levels: list[str],
    first_condition: str,
    second_condition: str,
    outcome: str,
) -> dict[str, Any]:
    """Test one selected condition difference using the fitted model covariance."""
    np, _, scipy, _ = _load_dependencies()
    fixed_names = list(result.fe_params.index)
    contrast = np.zeros(len(fixed_names), dtype=float)
    reference = condition_levels[0]

    for condition, sign in ((first_condition, 1.0), (second_condition, -1.0)):
        if condition == reference:
            continue
        coefficient_name = f"C(condition)[T.{condition}]"
        if coefficient_name not in fixed_names:
            raise RuntimeError(
                f"Could not find the fitted coefficient for condition {condition!r} "
                f"in the {outcome.upper()} model."
            )
        contrast[fixed_names.index(coefficient_name)] += sign

    fixed_covariance = result.cov_params().loc[fixed_names, fixed_names].to_numpy(dtype=float)
    estimate = float(contrast @ result.fe_params.to_numpy(dtype=float))
    variance = float(contrast @ fixed_covariance @ contrast)
    if variance < 0 and math.isclose(variance, 0.0, abs_tol=1e-12):
        variance = 0.0
    if variance <= 0 or not math.isfinite(variance):
        raise RuntimeError(
            f"Invalid variance for the selected {outcome.upper()} condition contrast: {variance}"
        )
    standard_error = math.sqrt(variance)
    wald_z = estimate / standard_error
    p_value = float(2.0 * scipy.stats.norm.sf(abs(wald_z)))
    critical_value = float(scipy.stats.norm.ppf(0.975))

    return {
        "outcome": outcome.upper(),
        "test": "Selected two-sided Wald contrast",
        "first_condition": first_condition,
        "second_condition": second_condition,
        "contrast_direction": "first_condition minus second_condition",
        "estimated_difference": estimate,
        "estimated_difference_percentage_points": estimate * 100.0,
        "standard_error": standard_error,
        "ci_95_lower": estimate - critical_value * standard_error,
        "ci_95_upper": estimate + critical_value * standard_error,
        "wald_z": wald_z,
        "wald_chi_square": wald_z**2,
        "degrees_of_freedom": 1,
        "p_value": p_value,
        "alpha": 0.05,
        "statistically_significant": p_value < 0.05,
    }


def _fit_outcome_with_pair(
    data: Any,
    outcome: str,
    compare_conditions: tuple[str, str] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    np, _, scipy, statsmodels_objects = _load_dependencies()
    statsmodels, smf = statsmodels_objects
    formula = f"{outcome} ~ C(condition) + C(speaker_id)"
    model = smf.mixedlm(
        formula,
        data,
        groups=data["all_rows"],
        re_formula="0",
        vc_formula={
            "listener": "0 + C(listener_id)",
            "utterance": "0 + C(utterance_id)",
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(
            reml=True,
            method=["lbfgs", "powell"],
            maxiter=2000,
            disp=False,
        )

    fixed_names = list(result.fe_params.index)
    condition_names = [name for name in fixed_names if name.startswith("C(condition)")]
    expected_df = len(data["condition"].cat.categories) - 1
    if len(condition_names) != expected_df:
        raise RuntimeError(
            f"Expected {expected_df} condition coefficients for {outcome}; "
            f"found {len(condition_names)}."
        )
    beta = result.fe_params.loc[condition_names].to_numpy(dtype=float)
    covariance = result.cov_params().loc[condition_names, condition_names].to_numpy(dtype=float)
    if np.linalg.matrix_rank(covariance) != expected_df:
        raise RuntimeError(f"Condition covariance matrix is singular for {outcome}.")
    statistic = float(beta @ np.linalg.solve(covariance, beta))
    if statistic < 0 and math.isclose(statistic, 0.0, abs_tol=1e-10):
        statistic = 0.0
    if statistic < 0 or not math.isfinite(statistic):
        raise RuntimeError(f"Invalid condition Wald statistic for {outcome}: {statistic}")
    p_value = float(scipy.stats.chi2.sf(statistic, expected_df))
    converged = bool(result.converged)
    warning_messages = sorted({str(item.message) for item in caught})
    residuals = np.asarray(result.resid, dtype=float)
    residual_sd_by_condition = {
        str(condition): float(values.std(ddof=1))
        for condition, values in data.assign(_residual=residuals)
        .groupby("condition", observed=True)["_residual"]
    }

    test_row = {
        "outcome": outcome.upper(),
        "test": "Omnibus joint Wald chi-square test for condition",
        "n_responses": int(len(data)),
        "n_listeners": int(data["listener_id"].nunique()),
        "n_utterances": int(data["utterance_id"].nunique()),
        "n_speakers": int(data["speaker_id"].nunique()),
        "n_conditions": int(data["condition"].nunique()),
        "wald_chi_square": statistic,
        "degrees_of_freedom": expected_df,
        "p_value": p_value,
        "alpha": 0.05,
        "statistically_significant": p_value < 0.05,
        "model_converged": converged,
    }
    audit = {
        "outcome": outcome.upper(),
        "formula": formula,
        "fit_method": "REML linear mixed model",
        "condition_test": "Joint Wald chi-square test of all condition coefficients",
        "condition_reference_level": str(data["condition"].cat.categories[0]),
        "speaker_fixed_effect": True,
        "listener_random_intercept": True,
        "utterance_random_intercept": True,
        "converged": converged,
        "warnings": warning_messages,
        "fixed_effect_names": fixed_names,
        "variance_component_names": list(model.exog_vc.names),
        "variance_component_estimates": [float(value) for value in result.vcomp],
        "residual_variance": float(result.scale),
        "outcome_zero_count": int((data[outcome] == 0).sum()),
        "outcome_above_one_count": int((data[outcome] > 1).sum()),
        "outcome_maximum": float(data[outcome].max()),
        "residual_skewness": float(scipy.stats.skew(residuals, bias=False)),
        "residual_excess_kurtosis": float(
            scipy.stats.kurtosis(residuals, bias=False)
        ),
        "residual_sd_by_condition": residual_sd_by_condition,
        "statsmodels_version": statsmodels.__version__,
    }
    pairwise_row = None
    if compare_conditions is not None:
        pairwise_row = _condition_contrast(
            result,
            [str(value) for value in data["condition"].cat.categories],
            compare_conditions[0],
            compare_conditions[1],
            outcome,
        )
    return test_row, audit, pairwise_row


def _fit_outcome(data: Any, outcome: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit one outcome model and return its omnibus condition test and audit."""
    test_row, audit, _ = _fit_outcome_with_pair(data, outcome, None)
    return test_row, audit


def _format_p_value(value: float) -> str:
    return "< .001" if value < 0.001 else f"= {value:.3f}"


def run_mixed_effects(
    scores_path: Path,
    output_root: Path,
    config_path: Path,
    compare_conditions: tuple[str, str] | None = None,
) -> list[dict[str, Any]]:
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file does not exist: {scores_path}")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {output_root}. Use a new directory for each run."
        )
    config = load_config(config_path)
    expected_conditions = [str(value) for value in config.get("expected_conditions", [])]
    if len(expected_conditions) < 2:
        raise ValueError("Configuration must define at least two expected conditions.")
    if compare_conditions is not None:
        first_condition, second_condition = compare_conditions
        if first_condition == second_condition:
            raise ValueError("The two selected conditions must be different.")
        unknown_conditions = [
            condition
            for condition in compare_conditions
            if condition not in expected_conditions
        ]
        if unknown_conditions:
            raise ValueError(
                "Unknown selected condition(s): "
                + ", ".join(unknown_conditions)
                + ". Choose from: "
                + ", ".join(expected_conditions)
            )

    data = _prepare_data(scores_path, expected_conditions)
    test_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for outcome in ("wer", "cer"):
        test_row, audit, pairwise_row = _fit_outcome_with_pair(
            data, outcome, compare_conditions
        )
        test_rows.append(test_row)
        audits.append(audit)
        if pairwise_row is not None:
            pairwise_rows.append(pairwise_row)

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "condition_effect_tests.csv", test_rows)
    if pairwise_rows:
        write_csv(output_root / "selected_condition_comparison.csv", pairwise_rows)
    summary_lines = [
        "Post-hoc linear mixed-effects omnibus condition tests",
        "",
    ]
    for row in test_rows:
        conclusion = "had" if row["statistically_significant"] else "did not have"
        summary_lines.append(
            f"Condition {conclusion} a statistically significant overall effect on "
            f"{row['outcome']} after adjusting for speaker and accounting for crossed "
            f"listener and utterance random intercepts, Wald chi-square({row['degrees_of_freedom']}) "
            f"= {row['wald_chi_square']:.3f}, p {_format_p_value(float(row['p_value']))}."
        )
    summary_lines.extend(
        [
            "",
            "These are omnibus tests: a significant result means that at least one condition differs, not that every condition differs.",
            "Both models converged. Distribution and residual diagnostics are recorded in model_audit.json and should be reviewed before formal reporting.",
        ]
    )
    if pairwise_rows:
        summary_lines.extend(["", "Selected condition comparison", ""])
        for row in pairwise_rows:
            conclusion = "was" if row["statistically_significant"] else "was not"
            summary_lines.append(
                f"For {row['outcome']}, {row['first_condition']} minus "
                f"{row['second_condition']} was {float(row['estimated_difference_percentage_points']):.2f} "
                f"percentage points (95% CI "
                f"{float(row['ci_95_lower']) * 100:.2f} to "
                f"{float(row['ci_95_upper']) * 100:.2f}); this selected difference "
                f"{conclusion} statistically significant, Wald z = "
                f"{float(row['wald_z']):.3f}, p {_format_p_value(float(row['p_value']))}."
            )
        summary_lines.append(
            "A positive difference means the first selected condition had a higher error rate."
        )
        summary_lines.append(
            "No multiplicity correction is applied because the output contains one selected comparison; "
            "choose the pair before inspecting its p-value."
        )
    (output_root / "condition_effect_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    audit_document = {
        "analysis": "Post-hoc linear mixed-effects omnibus condition tests",
        "scores_path": str(scores_path.resolve()),
        "scores_sha256": _sha256(scores_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "n_responses": int(len(data)),
        "n_listeners": int(data["listener_id"].nunique()),
        "n_utterances": int(data["utterance_id"].nunique()),
        "n_speakers": int(data["speaker_id"].nunique()),
        "condition_levels": expected_conditions,
        "selected_condition_comparison": list(compare_conditions)
        if compare_conditions is not None
        else None,
        "selected_comparison_multiplicity_adjustment": "none (one selected comparison)"
        if compare_conditions is not None
        else None,
        "models": audits,
    }
    (output_root / "model_audit.json").write_text(
        json.dumps(audit_document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return test_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the omnibus condition test and, optionally, one selected condition "
            "comparison using linear mixed-effects models."
        )
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("analysis_output/03_scores/transcription_scores.csv"),
        help="Response-level scored transcription CSV.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis_output/07_mixed_effects"),
        help="New directory for mixed-effects outputs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/analysis_config.json"),
        help="Analysis configuration containing the expected condition levels.",
    )
    parser.add_argument(
        "--compare-conditions",
        "--compare-condition",
        dest="compare_conditions",
        nargs=2,
        metavar=("FIRST", "SECOND"),
        help=(
            "Optionally test one selected condition pair in both WER and CER models. "
            "The reported difference is FIRST minus SECOND."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selected_pair = tuple(args.compare_conditions) if args.compare_conditions else None
        rows = run_mixed_effects(
            args.scores,
            args.output_root,
            args.config,
            compare_conditions=selected_pair,
        )
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for row in rows:
        print(
            f"{row['outcome']}: Wald chi-square({row['degrees_of_freedom']}) "
            f"= {row['wald_chi_square']:.3f}, p {_format_p_value(float(row['p_value']))}"
        )
    if selected_pair is not None:
        for row in read_csv(args.output_root / "selected_condition_comparison.csv"):
            print(
                f"{row['outcome']} selected comparison ({row['first_condition']} minus "
                f"{row['second_condition']}): {float(row['estimated_difference_percentage_points']):.2f} "
                f"percentage points, Wald z = {float(row['wald_z']):.3f}, "
                f"p {_format_p_value(float(row['p_value']))}"
            )
    print(f"Outputs: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
