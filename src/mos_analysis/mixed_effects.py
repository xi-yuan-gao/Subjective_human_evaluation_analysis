from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .analysis import load_config


REQUIRED_COLUMNS = {
    "participant_id",
    "session_id",
    "item_id",
    "condition",
    "speaker_id",
    "mos_rating",
}


def _load_dependencies() -> tuple[Any, Any, Any, Any]:
    """Load the optional numerical packages with one actionable error."""
    try:
        import numpy as np
        import pandas as pd
        import scipy
        import statsmodels
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise RuntimeError(
            "Mixed-effects dependencies are not installed. Install them with: "
            "python -m pip install -e '.[mixed-effects]'"
        ) from exc
    return np, pd, scipy, (statsmodels, smf)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [
            {str(key).strip(): (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in materialized:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_data(input_path: Path, expected_conditions: list[str]) -> Any:
    """Validate the analysis-ready MOS file and construct model identifiers."""
    _, pd, _, _ = _load_dependencies()
    rows = _read_csv(input_path)
    if not rows:
        raise ValueError(f"MOS input is empty: {input_path}")
    missing = sorted(REQUIRED_COLUMNS - set(rows[0]))
    if missing:
        raise ValueError("MOS input is missing required columns: " + ", ".join(missing))

    data = pd.DataFrame(rows)
    identifier_columns = (
        "participant_id",
        "session_id",
        "item_id",
        "condition",
        "speaker_id",
    )
    for column in identifier_columns:
        data[column] = data[column].astype(str).str.strip()
        if (data[column] == "").any():
            raise ValueError(f"MOS input contains blank {column} values.")
    data["mos_rating"] = pd.to_numeric(data["mos_rating"], errors="coerce")
    if data["mos_rating"].isna().any():
        raise ValueError("MOS input contains missing or non-numeric ratings.")
    if (~data["mos_rating"].isin([1, 2, 3, 4, 5])).any():
        raise ValueError("MOS ratings must be whole-number values from 1 through 5.")

    observed = set(data["condition"])
    expected = set(expected_conditions)
    if observed != expected:
        missing_conditions = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        details: list[str] = []
        if missing_conditions:
            details.append("missing: " + ", ".join(missing_conditions))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ValueError(
            "Condition levels do not match the configuration ("
            + "; ".join(details)
            + ")."
        )

    data["condition"] = pd.Categorical(
        data["condition"], categories=expected_conditions, ordered=True
    )
    data["listener_id"] = data["participant_id"] + "::" + data["session_id"]
    data["all_rows"] = "all"
    return data


def _condition_vector(
    fixed_names: list[str],
    condition_levels: list[str],
    speaker_levels: list[str],
    condition: str,
) -> Any:
    """Construct a fixed-effect vector averaged equally across speakers."""
    np, _, _, _ = _load_dependencies()
    vector = np.zeros(len(fixed_names), dtype=float)
    if "Intercept" in fixed_names:
        vector[fixed_names.index("Intercept")] = 1.0
    reference_condition = condition_levels[0]
    if condition != reference_condition:
        name = f"C(condition)[T.{condition}]"
        if name not in fixed_names:
            raise RuntimeError(f"Missing fitted condition coefficient: {name}")
        vector[fixed_names.index(name)] = 1.0
    for speaker in speaker_levels[1:]:
        name = f"C(speaker_id)[T.{speaker}]"
        if name not in fixed_names:
            raise RuntimeError(f"Missing fitted speaker coefficient: {name}")
        vector[fixed_names.index(name)] = 1.0 / len(speaker_levels)
    return vector


def _wald_result(
    result: Any,
    vector: Any,
    *,
    scipy: Any,
) -> dict[str, float]:
    fixed_names = list(result.fe_params.index)
    covariance = result.cov_params().loc[fixed_names, fixed_names].to_numpy(dtype=float)
    estimate = float(vector @ result.fe_params.to_numpy(dtype=float))
    variance = float(vector @ covariance @ vector)
    if variance <= 0 or not math.isfinite(variance):
        raise RuntimeError(f"Invalid fixed-effect contrast variance: {variance}")
    standard_error = math.sqrt(variance)
    z_value = estimate / standard_error
    critical = float(scipy.stats.norm.ppf(0.975))
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_95_lower": estimate - critical * standard_error,
        "ci_95_upper": estimate + critical * standard_error,
        "wald_z": z_value,
        "p_value": float(2.0 * scipy.stats.norm.sf(abs(z_value))),
    }


def _fit_model(
    data: Any,
    compare_conditions: tuple[str, str] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    """Fit the MOS linear mixed model and derive focused result tables."""
    np, _, scipy, statsmodels_objects = _load_dependencies()
    statsmodels, smf = statsmodels_objects
    formula = "mos_rating ~ C(condition) + C(speaker_id)"
    model = smf.mixedlm(
        formula,
        data,
        groups=data["all_rows"],
        re_formula="0",
        vc_formula={
            "item": "0 + C(item_id)",
            "listener": "0 + C(listener_id)",
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

    condition_levels = [str(value) for value in data["condition"].cat.categories]
    speaker_levels = sorted(str(value) for value in data["speaker_id"].unique())
    fixed_names = list(result.fe_params.index)
    condition_names = [name for name in fixed_names if name.startswith("C(condition)")]
    degrees_of_freedom = len(condition_levels) - 1
    if len(condition_names) != degrees_of_freedom:
        raise RuntimeError(
            f"Expected {degrees_of_freedom} condition coefficients; found {len(condition_names)}."
        )
    beta = result.fe_params.loc[condition_names].to_numpy(dtype=float)
    condition_covariance = result.cov_params().loc[
        condition_names, condition_names
    ].to_numpy(dtype=float)
    if np.linalg.matrix_rank(condition_covariance) != degrees_of_freedom:
        raise RuntimeError("Condition covariance matrix is singular.")
    statistic = float(beta @ np.linalg.solve(condition_covariance, beta))
    if statistic < 0 and math.isclose(statistic, 0.0, abs_tol=1e-10):
        statistic = 0.0
    if statistic < 0 or not math.isfinite(statistic):
        raise RuntimeError(f"Invalid condition Wald statistic: {statistic}")
    p_value = float(scipy.stats.chi2.sf(statistic, degrees_of_freedom))

    condition_test = {
        "outcome": "MOS rating",
        "test": "Omnibus joint Wald chi-square test for condition",
        "n_responses": int(len(data)),
        "n_listeners": int(data["listener_id"].nunique()),
        "n_items": int(data["item_id"].nunique()),
        "n_speakers": int(data["speaker_id"].nunique()),
        "n_conditions": len(condition_levels),
        "wald_chi_square": statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "alpha": 0.05,
        "statistically_significant": p_value < 0.05,
        "model_converged": bool(result.converged),
    }

    fixed_covariance = result.cov_params().loc[fixed_names, fixed_names]
    fixed_rows: list[dict[str, Any]] = []
    critical = float(scipy.stats.norm.ppf(0.975))
    for term in fixed_names:
        estimate = float(result.fe_params[term])
        standard_error = math.sqrt(float(fixed_covariance.loc[term, term]))
        z_value = estimate / standard_error
        fixed_rows.append(
            {
                "term": term,
                "estimate": estimate,
                "standard_error": standard_error,
                "ci_95_lower": estimate - critical * standard_error,
                "ci_95_upper": estimate + critical * standard_error,
                "wald_z": z_value,
                "p_value": float(2.0 * scipy.stats.norm.sf(abs(z_value))),
            }
        )

    adjusted_rows: list[dict[str, Any]] = []
    vectors: dict[str, Any] = {}
    for condition in condition_levels:
        vector = _condition_vector(
            fixed_names, condition_levels, speaker_levels, condition
        )
        vectors[condition] = vector
        estimate = _wald_result(result, vector, scipy=scipy)
        adjusted_rows.append(
            {
                "condition": condition,
                "adjusted_mean_mos": estimate["estimate"],
                "standard_error": estimate["standard_error"],
                "ci_95_lower": estimate["ci_95_lower"],
                "ci_95_upper": estimate["ci_95_upper"],
                "speaker_adjustment": "equal average across observed speaker levels",
            }
        )

    pairwise_row: dict[str, Any] | None = None
    if compare_conditions is not None:
        first, second = compare_conditions
        contrast = vectors[first] - vectors[second]
        comparison = _wald_result(result, contrast, scipy=scipy)
        pairwise_row = {
            "first_condition": first,
            "second_condition": second,
            "contrast_direction": "first_condition minus second_condition",
            "estimated_difference_mos_points": comparison["estimate"],
            "standard_error": comparison["standard_error"],
            "ci_95_lower": comparison["ci_95_lower"],
            "ci_95_upper": comparison["ci_95_upper"],
            "wald_z": comparison["wald_z"],
            "wald_chi_square": comparison["wald_z"] ** 2,
            "degrees_of_freedom": 1,
            "p_value": comparison["p_value"],
            "alpha": 0.05,
            "statistically_significant": comparison["p_value"] < 0.05,
            "multiplicity_adjustment": "none (one comparison selected before testing)",
        }

    residuals = np.asarray(result.resid, dtype=float)
    variance_components = {
        name: float(value)
        for name, value in zip(model.exog_vc.names, result.vcomp)
    }
    audit = {
        "model_family": "Gaussian linear mixed-effects model",
        "formula": formula,
        "fit_method": "REML",
        "condition_reference_level": condition_levels[0],
        "speaker_reference_level": speaker_levels[0],
        "speaker_fixed_effect": True,
        "listener_random_intercept": True,
        "item_random_intercept": True,
        "converged": bool(result.converged),
        "warnings": sorted({str(item.message) for item in caught}),
        "fixed_effect_names": fixed_names,
        "variance_components": variance_components,
        "residual_variance": float(result.scale),
        "residual_skewness": float(scipy.stats.skew(residuals, bias=False)),
        "residual_excess_kurtosis": float(
            scipy.stats.kurtosis(residuals, bias=False)
        ),
        "rating_minimum": float(data["mos_rating"].min()),
        "rating_maximum": float(data["mos_rating"].max()),
        "statsmodels_version": statsmodels.__version__,
    }
    return condition_test, fixed_rows, adjusted_rows, audit, pairwise_row


def _format_p_value(value: float) -> str:
    return "< .001" if value < 0.001 else f"= {value:.3f}"


def run_mos_mixed_effects(
    input_path: Path,
    output_root: Path,
    config_path: Path,
    compare_conditions: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Run the focused MOS condition-effect mixed analysis."""
    if not input_path.is_file():
        raise FileNotFoundError(f"MOS input does not exist: {input_path}")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {output_root}. Use a new directory."
        )
    config = load_config(config_path)
    conditions = [str(value) for value in config.get("expected_conditions", [])]
    if len(conditions) < 2:
        raise ValueError("Configuration must define at least two conditions.")
    if compare_conditions is not None:
        first, second = compare_conditions
        if first == second:
            raise ValueError("The selected conditions must be different.")
        unknown = [value for value in compare_conditions if value not in conditions]
        if unknown:
            raise ValueError(
                "Unknown selected condition(s): "
                + ", ".join(unknown)
                + ". Choose from: "
                + ", ".join(conditions)
            )

    data = _prepare_data(input_path, conditions)
    condition_test, fixed_rows, adjusted_rows, audit, pairwise_row = _fit_model(
        data, compare_conditions
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "condition_effect_test.csv", [condition_test])
    _write_csv(output_root / "fixed_effect_estimates.csv", fixed_rows)
    _write_csv(output_root / "adjusted_condition_means.csv", adjusted_rows)
    if pairwise_row is not None:
        _write_csv(output_root / "selected_condition_comparison.csv", [pairwise_row])

    conclusion = "had" if condition_test["statistically_significant"] else "did not have"
    lines = [
        "MOS linear mixed-effects analysis",
        "",
        (
            f"Condition {conclusion} a statistically significant overall effect on MOS "
            f"after adjusting for speaker and accounting for crossed listener and item "
            f"random intercepts, Wald chi-square({condition_test['degrees_of_freedom']}) "
            f"= {condition_test['wald_chi_square']:.3f}, "
            f"p {_format_p_value(float(condition_test['p_value']))}."
        ),
        "A significant omnibus result means at least one condition differs; it does not mean every pair differs.",
        "The Gaussian model treats the 1-5 MOS scale as approximately continuous.",
    ]
    if pairwise_row is not None:
        significant = "was" if pairwise_row["statistically_significant"] else "was not"
        lines.extend(
            [
                "",
                (
                    f"Selected comparison: {pairwise_row['first_condition']} minus "
                    f"{pairwise_row['second_condition']} = "
                    f"{pairwise_row['estimated_difference_mos_points']:.3f} MOS points "
                    f"(95% CI {pairwise_row['ci_95_lower']:.3f} to "
                    f"{pairwise_row['ci_95_upper']:.3f}), Wald z = "
                    f"{pairwise_row['wald_z']:.3f}, "
                    f"p {_format_p_value(float(pairwise_row['p_value']))}; the difference "
                    f"{significant} statistically significant."
                ),
                "A positive estimate means the first condition has the higher adjusted MOS.",
            ]
        )
    (output_root / "condition_effect_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    audit_document = {
        "analysis": "MOS linear mixed-effects condition test",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path.resolve()),
        "input_sha256": _sha256(input_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "n_responses": int(len(data)),
        "n_listeners": int(data["listener_id"].nunique()),
        "n_items": int(data["item_id"].nunique()),
        "n_speakers": int(data["speaker_id"].nunique()),
        "condition_levels": conditions,
        "selected_condition_comparison": (
            list(compare_conditions) if compare_conditions is not None else None
        ),
        "model": audit,
    }
    (output_root / "model_audit.json").write_text(
        json.dumps(audit_document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return condition_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test the overall MOS condition effect with a linear mixed model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Validated MOS analysis-ready CSV.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New or empty directory for mixed-effects outputs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/mos_analysis_config.json"),
        help="MOS configuration containing the condition order.",
    )
    parser.add_argument(
        "--compare-conditions",
        nargs=2,
        metavar=("FIRST", "SECOND"),
        help="Optionally test one selected condition pair as FIRST minus SECOND.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selected = tuple(args.compare_conditions) if args.compare_conditions else None
        result = run_mos_mixed_effects(
            args.input,
            args.output_root,
            args.config,
            compare_conditions=selected,
        )
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"MOS: Wald chi-square({result['degrees_of_freedom']}) = "
        f"{result['wald_chi_square']:.3f}, "
        f"p {_format_p_value(float(result['p_value']))}"
    )
    print(f"Outputs: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
