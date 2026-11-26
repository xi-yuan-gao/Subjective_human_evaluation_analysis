"""Count-based mixed-effects analysis of transcription WER and CER.

Word and character error counts are modeled with negative binomial generalized
linear mixed models (glmmTMB, run in R), with the reference length as an offset,
so that exponentiated condition effects are ratios of error rates.

Pre-specified analysis plan
---------------------------
* Outcomes: WER (primary) and CER (secondary), as error counts with
  log(reference length) offsets.
* Condition coded +0.5 (first condition) / -0.5 (second condition).
* Random effects: by-listener intercepts; by-utterance intercepts and condition
  slopes. If that model is singular or does not converge, the correlation is
  dropped, then the slope (intercepts only).
* Family: negative binomial (nbinom2). Poisson reported as a sensitivity model.
* Condition test: likelihood ratio test (1 df). Wald CIs for rate ratios.
* Robustness: utterance-level paired Wilcoxon signed-rank test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
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
    "word_errors",
    "reference_words",
    "character_errors",
    "reference_characters",
}

OUTCOMES = {
    "WER": {"errors": "word_errors", "denominator": "reference_words", "rate": "wer"},
    "CER": {"errors": "character_errors", "denominator": "reference_characters", "rate": "cer"},
}
PRIMARY_OUTCOME = "WER"

R_SCRIPT = Path(__file__).resolve().parent / "glmm_count_models.R"
DEFAULT_SEED = 20261108
DEFAULT_SIMULATIONS = 1000
Z_975 = 1.959963984540054


def _load_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import pandas as pd
        import scipy.stats as stats
    except ImportError as exc:
        raise RuntimeError(
            "Mixed-effects dependencies are not installed. Install the optional "
            "extra with: python -m pip install -e '.[mixed-effects]'"
        ) from exc
    return np, pd, stats


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_p_value(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "= n/a"
    return "< .001" if value < 0.001 else f"= {value:.3f}"


def _find_rscript() -> str:
    candidate = os.environ.get("TRANSCRIPTION_RSCRIPT") or shutil.which("Rscript")
    if not candidate:
        raise RuntimeError(
            "Rscript was not found. This analysis fits negative binomial mixed models "
            "with the R package glmmTMB. Install R (https://cran.r-project.org), then in R "
            "run install.packages('glmmTMB'). If Rscript is installed but not on PATH, "
            "set the environment variable TRANSCRIPTION_RSCRIPT to its full path."
        )
    return candidate


# ---------------------------------------------------------------------------
# Data preparation and design checks
# ---------------------------------------------------------------------------


def _prepare_data(scores_path: Path, condition_order: list[str]) -> Any:
    np, pd, _ = _load_dependencies()
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

    for spec in OUTCOMES.values():
        for column in (spec["errors"], spec["denominator"]):
            values = pd.to_numeric(data[column], errors="coerce")
            if values.isna().any():
                raise ValueError(f"Scores file contains missing or non-numeric {column} values.")
            if (values < 0).any() or not np.allclose(values, values.round()):
                raise ValueError(f"{column} must contain non-negative whole numbers.")
            data[column] = values.round().astype(int)
        if (data[spec["denominator"]] <= 0).any():
            raise ValueError(f"{spec['denominator']} must be positive for every row.")
        rate = pd.to_numeric(data[spec["rate"]], errors="coerce")
        recomputed = data[spec["errors"]] / data[spec["denominator"]]
        if rate.isna().any() or not np.allclose(rate, recomputed, atol=1e-6):
            raise ValueError(
                f"{spec['rate']} does not equal {spec['errors']} / {spec['denominator']} "
                "for every row; counts and rates appear to come from different texts."
            )
        data[spec["rate"]] = rate

    if "is_attention_check" in data.columns:
        flags = data["is_attention_check"].astype(str).str.strip().str.lower()
        if flags.isin(["true", "1", "yes"]).any():
            raise ValueError("Scores file contains attention-check rows; remove them first.")

    observed = set(data["condition"])
    expected = set(condition_order)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ValueError("Condition levels do not match the configuration (" + "; ".join(details) + ").")

    data["condition"] = pd.Categorical(data["condition"], categories=condition_order, ordered=True)
    data["listener_id"] = data["participant_id"] + "::" + data["session_id"]
    data["model_c"] = np.where(data["condition"] == condition_order[0], 0.5, -0.5)
    return data


def _design_checks(data: Any, condition_order: list[str]) -> dict[str, Any]:
    """Verify the counterbalanced design; raise on structural errors."""
    _, pd, _ = _load_dependencies()
    per_utterance = pd.crosstab(data["utterance_id"], data["condition"])
    per_listener = pd.crosstab(data["listener_id"], data["condition"])

    speakers_per_utterance = data.groupby("utterance_id")["speaker_id"].nunique()
    if (speakers_per_utterance > 1).any():
        bad = speakers_per_utterance[speakers_per_utterance > 1].index.tolist()
        raise ValueError(f"Utterance IDs mapped to more than one speaker: {bad[:5]}")
    for spec in OUTCOMES.values():
        n_lengths = data.groupby("utterance_id")[spec["denominator"]].nunique()
        if (n_lengths > 1).any():
            bad = n_lengths[n_lengths > 1].index.tolist()
            raise ValueError(
                f"Utterances with different {spec['denominator']} across rows "
                f"(ID pairing or reference text problem): {bad[:5]}"
            )
    duplicates = data.duplicated(["listener_id", "utterance_id"]).sum()
    if duplicates:
        raise ValueError(f"{duplicates} listener heard the same utterance more than once.")

    unpaired = per_utterance.index[(per_utterance == 0).any(axis=1)].tolist()
    single_condition_listeners = per_listener.index[(per_listener == 0).any(axis=1)].tolist()
    return {
        "n_utterances": int(per_utterance.shape[0]),
        "utterances_observed_in_both_conditions": int(per_utterance.shape[0] - len(unpaired)),
        "utterances_missing_a_condition": unpaired,
        "observations_per_utterance_condition": sorted(
            {int(value) for value in per_utterance.to_numpy().ravel()}
        ),
        "listeners_hearing_both_conditions": int(per_listener.shape[0] - len(single_condition_listeners)),
        "listeners_hearing_one_condition": single_condition_listeners,
        "trials_per_listener_condition": sorted(
            {int(value) for value in per_listener.to_numpy().ravel()}
        ),
        "fully_paired": not unpaired and not single_condition_listeners,
    }


# ---------------------------------------------------------------------------
# Descriptive statistics and robustness check
# ---------------------------------------------------------------------------


def _corpus_rate(frame: Any, spec: dict[str, str]) -> float:
    return float(frame[spec["errors"]].sum() / frame[spec["denominator"]].sum())


def _descriptives(data: Any) -> tuple[list[dict], list[dict], list[dict]]:
    overall, by_listener, by_speaker = [], [], []
    for outcome, spec in OUTCOMES.items():
        for condition, group in data.groupby("condition", observed=True):
            counts = group[spec["errors"]]
            overall.append({
                "outcome": outcome,
                "condition": str(condition),
                "n_transcriptions": int(len(group)),
                "total_errors": int(counts.sum()),
                "total_reference_units": int(group[spec["denominator"]].sum()),
                "corpus_rate": _corpus_rate(group, spec),
                "mean_utterance_rate": float(group[spec["rate"]].mean()),
                "share_perfect_transcriptions": float((counts == 0).mean()),
                "error_count_mean": float(counts.mean()),
                "error_count_variance": float(counts.var(ddof=1)),
            })
        for (listener, condition), group in data.groupby(["listener_id", "condition"], observed=True):
            by_listener.append({
                "outcome": outcome, "listener_id": listener, "condition": str(condition),
                "n_transcriptions": int(len(group)), "corpus_rate": _corpus_rate(group, spec),
            })
        for (speaker, condition), group in data.groupby(["speaker_id", "condition"], observed=True):
            by_speaker.append({
                "outcome": outcome, "speaker_id": speaker, "condition": str(condition),
                "n_transcriptions": int(len(group)), "corpus_rate": _corpus_rate(group, spec),
            })
    return overall, by_listener, by_speaker


def _utterance_wilcoxon(data: Any, first: str, second: str) -> list[dict]:
    """Paired Wilcoxon signed-rank test across utterances (pooled over listeners)."""
    np, _, stats = _load_dependencies()
    rows = []
    for outcome, spec in OUTCOMES.items():
        pooled = data.groupby(["utterance_id", "condition"], observed=True)[
            [spec["errors"], spec["denominator"]]
        ].sum()
        rates = (pooled[spec["errors"]] / pooled[spec["denominator"]]).unstack("condition")
        rates = rates.dropna(subset=[first, second])
        differences = rates[first] - rates[second]
        nonzero = differences[differences != 0]
        if len(nonzero) >= 1:
            result = stats.wilcoxon(rates[first], rates[second], zero_method="wilcox")
            statistic, p_value = float(result.statistic), float(result.pvalue)
        else:
            statistic, p_value = float("nan"), float("nan")
        ranks = stats.rankdata(np.abs(nonzero))
        w_plus = float(ranks[nonzero.to_numpy() > 0].sum())
        w_minus = float(ranks[nonzero.to_numpy() < 0].sum())
        total = w_plus + w_minus
        rows.append({
            "outcome": outcome,
            "first_condition": first,
            "second_condition": second,
            "n_utterances": int(len(differences)),
            "n_nonzero_differences": int(len(nonzero)),
            "n_first_higher": int((differences > 0).sum()),
            "n_second_higher": int((differences < 0).sum()),
            "median_difference_first_minus_second": float(differences.median()),
            "w_plus": w_plus,
            "w_minus": w_minus,
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
            "rank_biserial_r": (w_plus - w_minus) / total if total > 0 else float("nan"),
        })
    return rows


# ---------------------------------------------------------------------------
# R backend
# ---------------------------------------------------------------------------


def _run_r_models(data: Any, work_dir: Path, seed: int, n_simulations: int) -> dict[str, Any]:
    _, pd, _ = _load_dependencies()
    if not R_SCRIPT.exists():
        raise FileNotFoundError(f"R script not found next to this module: {R_SCRIPT}")
    rscript = _find_rscript()
    work_dir.mkdir(parents=True, exist_ok=True)
    columns = ["listener_id", "utterance_id", "model_c"] + [
        column for spec in OUTCOMES.values() for column in (spec["errors"], spec["denominator"])
    ]
    input_path = work_dir / "model_input.csv"
    data[columns].to_csv(input_path, index=False)

    completed = subprocess.run(
        [rscript, "--vanilla", str(R_SCRIPT), str(input_path), str(work_dir),
         str(seed), str(n_simulations)],
        capture_output=True, text=True,
    )
    (work_dir / "r_log.txt").write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "The R model fitting failed. Last lines of the R output:\n"
            + "\n".join(completed.stderr.strip().splitlines()[-15:])
        )

    def load(name: str) -> Any:
        return pd.read_csv(work_dir / name, keep_default_na=True)

    return {
        "results": load("results.csv"),
        "attempts": load("attempts.csv").fillna(""),
        "random_effects": load("random_effects.csv"),
        "diagnostics": load("diagnostics.csv"),
        "versions": (work_dir / "versions.txt").read_text(encoding="utf-8").splitlines(),
    }


def _effect_rows(results: Any, first: str, second: str) -> list[dict[str, Any]]:
    """Convert log-scale coefficients into rate ratios and per-condition rates."""
    rows = []
    for record in results.to_dict("records"):
        b0, b1 = float(record["intercept"]), float(record["condition_coef"])
        v00, v11 = float(record["var_intercept"]), float(record["var_condition"])
        v01 = float(record["cov_intercept_condition"])
        se1 = math.sqrt(v11)

        def typical_rate(sign: float) -> tuple[float, float, float]:
            log_rate = b0 + sign * 0.5 * b1
            se = math.sqrt(max(v00 + 0.25 * v11 + sign * v01, 0.0))
            return (math.exp(log_rate), math.exp(log_rate - Z_975 * se),
                    math.exp(log_rate + Z_975 * se))

        rate_first = typical_rate(1.0)
        rate_second = typical_rate(-1.0)
        wald_z = b1 / se1
        lrt_p = record.get("lrt_p_value")
        rows.append({
            "outcome": record["outcome"],
            "role": record["role"],
            "family": record["family"],
            "utterance_random_effects": record["utterance_random_effects"],
            "first_condition": first,
            "second_condition": second,
            "n_transcriptions": int(record["n_obs"]),
            "rate_ratio_first_over_second": math.exp(b1),
            "rate_ratio_ci_95_lower": math.exp(b1 - Z_975 * se1),
            "rate_ratio_ci_95_upper": math.exp(b1 + Z_975 * se1),
            "percent_change_first_vs_second": (math.exp(b1) - 1.0) * 100.0,
            "log_rate_ratio": b1,
            "log_rate_ratio_se": se1,
            "wald_z": wald_z,
            "wald_p_value": float(2.0 * _load_dependencies()[2].norm.sf(abs(wald_z))),
            "lrt_chi_square": float(record["lrt_chi_square"]),
            "lrt_df": int(record["lrt_df"]),
            "lrt_p_value": float(lrt_p) if lrt_p is not None else float("nan"),
            "typical_rate_first": rate_first[0],
            "typical_rate_first_ci_95_lower": rate_first[1],
            "typical_rate_first_ci_95_upper": rate_first[2],
            "typical_rate_second": rate_second[0],
            "typical_rate_second_ci_95_lower": rate_second[1],
            "typical_rate_second_ci_95_upper": rate_second[2],
            "nb_theta": float(record["nb_theta"]),
            "aic": float(record["aic"]),
            "fit_accepted": bool(record["fit_accepted"]),
            "fit_note": record["fit_note"],
            "warnings": "" if str(record.get("warnings")) == "nan" else record.get("warnings"),
        })
    return rows


# ---------------------------------------------------------------------------
# Report text
# ---------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _summary_text(
    primary: dict[str, dict], sensitivity: list[dict], descriptives: list[dict],
    diagnostics: list[dict], random_effects: list[dict], wilcoxon: list[dict],
    design: dict[str, Any], first: str, second: str,
) -> str:
    lines = [
        "Transcription error analysis: negative binomial generalized linear mixed models",
        "=" * 78,
        "",
        f"Comparison: {first} (first) vs {second} (second).",
        "Rate ratios are first / second: values below 1 mean the first condition had fewer errors.",
        f"Primary outcome: {PRIMARY_OUTCOME}. Secondary outcome: CER.",
        "",
        "Design checks",
        "-------------",
        f"Utterances observed in both conditions: {design['utterances_observed_in_both_conditions']} "
        f"of {design['n_utterances']}; observations per utterance x condition: "
        f"{design['observations_per_utterance_condition']}.",
        f"Listeners hearing both conditions: {design['listeners_hearing_both_conditions']}; "
        f"trials per listener x condition: {design['trials_per_listener_condition']}.",
    ]
    if not design["fully_paired"]:
        lines.append("WARNING: the design is not fully paired; see model_audit.json.")

    lines += ["", "Descriptive corpus-level rates (total errors / total reference units)", "-" * 70]
    for row in descriptives:
        lines.append(
            f"{row['outcome']} {row['condition']}: {_pct(row['corpus_rate'])} "
            f"({row['total_errors']} errors / {row['total_reference_units']}); "
            f"perfect transcriptions {_pct(row['share_perfect_transcriptions'])}; "
            f"error-count mean {row['error_count_mean']:.2f}, variance {row['error_count_variance']:.2f}."
        )

    lines += ["", "Model results", "-" * 13]
    for outcome in OUTCOMES:
        row = primary[outcome]
        sds = {
            (r["group"], r["term"]): r["value"]
            for r in random_effects if r["outcome"] == outcome and r["type"] == "sd"
        }
        structure = {
            "full": "by-utterance random intercepts and condition slopes (with correlation)",
            "diag": "by-utterance random intercepts and condition slopes (uncorrelated)",
            "intercept": "by-utterance random intercepts",
        }[row["utterance_random_effects"]]
        diag = next(d for d in diagnostics if d["outcome"] == outcome and d["role"] == "primary")
        direction = "fewer" if row["rate_ratio_first_over_second"] < 1 else "more"
        significant = row["lrt_p_value"] < 0.05
        lines += [
            "",
            f"{outcome} ({'primary' if outcome == PRIMARY_OUTCOME else 'secondary'})",
            f"  Model: negative binomial GLMM, offset log(reference length), {structure}, "
            f"by-listener random intercepts.",
            f"  Rate ratio ({first} / {second}) = {row['rate_ratio_first_over_second']:.2f}, "
            f"95% CI [{row['rate_ratio_ci_95_lower']:.2f}, {row['rate_ratio_ci_95_upper']:.2f}]; "
            f"LRT chi-square(1) = {row['lrt_chi_square']:.2f}, p {_format_p_value(row['lrt_p_value'])}.",
            f"  Typical-listener, typical-utterance {outcome}: {first} {_pct(row['typical_rate_first'])} "
            f"[{_pct(row['typical_rate_first_ci_95_lower'])}, {_pct(row['typical_rate_first_ci_95_upper'])}], "
            f"{second} {_pct(row['typical_rate_second'])} "
            f"[{_pct(row['typical_rate_second_ci_95_lower'])}, {_pct(row['typical_rate_second_ci_95_upper'])}].",
            "  Random-effect SDs (log scale): "
            + ", ".join(f"{group} {term} = {value:.2f}" for (group, term), value in sds.items())
            + f"; NB theta = {row['nb_theta']:.2f}.",
            f"  Distribution check ({diag['n_simulations']} simulated datasets): zero transcriptions "
            f"{diag['observed_zero_count']} observed vs {diag['mean_simulated_zero_count']:.1f} expected "
            f"(p {_format_p_value(diag['zero_p_value'])}); count variance {diag['observed_count_variance']:.2f} "
            f"vs {diag['mean_simulated_count_variance']:.2f} (p {_format_p_value(diag['variance_p_value'])}); "
            f"largest count {diag['observed_max_count']} vs {diag['mean_simulated_max_count']:.1f} "
            f"(p {_format_p_value(diag['max_p_value'])}).",
            "  Interpretation: "
            + (
                f"{first} produced {abs(row['percent_change_first_vs_second']):.0f}% {direction} errors "
                f"per reference unit than {second}, and the difference is statistically reliable."
                if significant else
                f"the point estimate suggests {abs(row['percent_change_first_vs_second']):.0f}% "
                f"{direction} errors for {first}, but the difference is not statistically reliable; "
                f"the CI is compatible with rate ratios from {row['rate_ratio_ci_95_lower']:.2f} "
                f"to {row['rate_ratio_ci_95_upper']:.2f}, so neither equivalence nor a difference is established."
            ),
        ]

    lines += ["", "Sensitivity analyses", "-" * 20]
    for row in sensitivity:
        label = {
            "sensitivity_poisson": "Poisson, same random effects",
            "sensitivity_intercept_only": "Negative binomial, utterance intercepts only",
        }.get(row["role"], row["role"])
        note = "" if row["fit_accepted"] else f" (fit flagged: {row['fit_note']})"
        lines.append(
            f"{row['outcome']} - {label}: rate ratio {row['rate_ratio_first_over_second']:.2f} "
            f"[{row['rate_ratio_ci_95_lower']:.2f}, {row['rate_ratio_ci_95_upper']:.2f}], "
            f"LRT p {_format_p_value(row['lrt_p_value'])}{note}."
        )
    for row in wilcoxon:
        lines.append(
            f"{row['outcome']} - paired Wilcoxon across utterances: {first} higher on "
            f"{row['n_first_higher']}, {second} higher on {row['n_second_higher']}, tied on "
            f"{row['n_utterances'] - row['n_nonzero_differences']}; "
            f"p {_format_p_value(row['p_value'])}, rank-biserial r = {row['rank_biserial_r']:.2f}."
        )

    lines += [
        "",
        "Notes",
        "-----",
        "Typical rates are model estimates for a listener and utterance with average random effects;",
        "they can differ from the descriptive corpus rates, which weight utterances by length.",
        "Speaker is not modeled separately: utterances are nested in speakers and four speakers are",
        "too few to estimate a speaker variance; generalization beyond these speakers is limited.",
        "Narrower CIs in the intercept-only sensitivity model indicate that the condition effect",
        "varies across utterances; the primary model accounts for this.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def run_mixed_effects(
    scores_path: Path,
    output_root: Path,
    config_path: Path,
    compare_conditions: tuple[str, str] | None = None,
) -> list[dict[str, Any]]:
    _, pd, _ = _load_dependencies()
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file does not exist: {scores_path}")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {output_root}. Use a new directory for each run."
        )
    config = load_config(config_path)
    expected_conditions = [str(value) for value in config.get("expected_conditions", [])]
    if len(expected_conditions) != 2:
        raise ValueError(
            "This count-model analysis currently supports exactly two conditions; "
            f"the configuration lists {len(expected_conditions)}."
        )
    if compare_conditions is not None:
        if compare_conditions[0] == compare_conditions[1]:
            raise ValueError("The two selected conditions must be different.")
        unknown = [c for c in compare_conditions if c not in expected_conditions]
        if unknown:
            raise ValueError(
                "Unknown selected condition(s): " + ", ".join(unknown)
                + ". Choose from: " + ", ".join(expected_conditions)
            )
        condition_order = list(compare_conditions)
    else:
        condition_order = expected_conditions
    first, second = condition_order

    seed = int(config.get("mixed_effects_seed", DEFAULT_SEED))
    n_simulations = int(config.get("mixed_effects_simulations", DEFAULT_SIMULATIONS))

    data = _prepare_data(scores_path, condition_order)
    design = _design_checks(data, condition_order)
    descriptives, by_listener, by_speaker = _descriptives(data)
    wilcoxon = _utterance_wilcoxon(data, first, second)

    output_root.mkdir(parents=True, exist_ok=True)
    r_output = _run_r_models(data, output_root / "r_work", seed, n_simulations)

    effect_rows = _effect_rows(r_output["results"], first, second)
    primary = {row["outcome"]: row for row in effect_rows if row["role"] == "primary"}
    sensitivity = [row for row in effect_rows if row["role"] != "primary"]
    diagnostics = r_output["diagnostics"].to_dict("records")
    random_effects = r_output["random_effects"].to_dict("records")

    primary_rows = [primary[outcome] for outcome in OUTCOMES]
    write_csv(output_root / "condition_effect_tests.csv", primary_rows)
    write_csv(output_root / "sensitivity_analyses.csv", sensitivity + [
        {"outcome": row["outcome"], "role": "sensitivity_utterance_wilcoxon", **row}
        for row in wilcoxon
    ])
    write_csv(output_root / "model_selection_attempts.csv", r_output["attempts"].to_dict("records"))
    write_csv(output_root / "random_effects.csv", random_effects)
    write_csv(output_root / "distribution_diagnostics.csv", diagnostics)
    write_csv(output_root / "descriptive_corpus_rates.csv", descriptives)
    write_csv(output_root / "descriptive_by_listener.csv", by_listener)
    write_csv(output_root / "descriptive_by_speaker.csv", by_speaker)

    (output_root / "condition_effect_summary.txt").write_text(
        _summary_text(primary, sensitivity, descriptives, diagnostics, random_effects,
                      wilcoxon, design, first, second),
        encoding="utf-8",
    )

    audit_document = {
        "analysis": "Negative binomial GLMMs of word and character error counts",
        "analysis_plan": [line.lstrip("* ").strip() for line in __doc__.split("Pre-specified analysis plan")[1].strip().splitlines()[1:] if line.strip()],
        "scores_path": str(scores_path.resolve()),
        "scores_sha256": _sha256(scores_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "r_versions": r_output["versions"],
        "random_seed": seed,
        "n_simulations": n_simulations,
        "condition_coding": {first: 0.5, second: -0.5},
        "rate_ratio_direction": f"{first} / {second}",
        "primary_outcome": PRIMARY_OUTCOME,
        "n_transcriptions": int(len(data)),
        "n_listeners": int(data["listener_id"].nunique()),
        "n_utterances": int(data["utterance_id"].nunique()),
        "n_speakers": int(data["speaker_id"].nunique()),
        "design_checks": design,
        "model_selection_attempts": r_output["attempts"].to_dict("records"),
        "models": effect_rows,
        "random_effects": random_effects,
        "distribution_diagnostics": diagnostics,
    }
    (output_root / "model_audit.json").write_text(
        json.dumps(audit_document, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return primary_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare WER and CER between two conditions with negative binomial "
            "generalized linear mixed models of error counts (glmmTMB via R)."
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
        default=Path("config/transcription_config.json"),
        help="Analysis configuration containing the expected condition levels.",
    )
    parser.add_argument(
        "--compare-conditions",
        "--compare-condition",
        dest="compare_conditions",
        nargs=2,
        metavar=("FIRST", "SECOND"),
        help=(
            "Optionally set the direction of the comparison: rate ratios are FIRST / SECOND. "
            "Defaults to the order in the configuration."
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
            f"{row['outcome']}: rate ratio ({row['first_condition']} / {row['second_condition']}) "
            f"= {row['rate_ratio_first_over_second']:.2f} "
            f"[{row['rate_ratio_ci_95_lower']:.2f}, {row['rate_ratio_ci_95_upper']:.2f}], "
            f"LRT chi-square(1) = {row['lrt_chi_square']:.2f}, "
            f"p {_format_p_value(row['lrt_p_value'])} "
            f"[{row['utterance_random_effects']} utterance random effects]"
        )
    print(f"Outputs: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
