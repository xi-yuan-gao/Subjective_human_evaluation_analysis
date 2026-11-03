"""Utterance- and model-level human/ASR agreement analysis."""
from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Expected a finite nonnegative rate, got {value!r}")
    return result


TIE_DECIMALS = 12  # scores equal to 12 decimals are ties (removes float noise such as 1/6 stored two ways)


def average_ranks(values: list[float]) -> list[float]:
    """Ascending ranks, with average ranks for ties (including repeated zeros)."""
    values = [round(v, TIE_DECIMALS) for v in values]
    indices = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indices):
        end = start + 1
        while end < len(indices) and values[indices[end]] == values[indices[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for i in indices[start:end]:
            ranks[i] = rank
        start = end
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y):
        raise ValueError("Paired vectors must have equal length")
    if len(x) < 3:
        return None
    mx, my = mean(x), mean(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    xx, yy = math.fsum(v * v for v in dx), math.fsum(v * v for v in dy)
    if xx == 0 or yy == 0:
        return None
    value = math.fsum(a * b for a, b in zip(dx, dy)) / math.sqrt(xx * yy)
    return max(-1.0, min(1.0, value))


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(average_ranks(x), average_ranks(y))


def correlation_status(x: list[float], y: list[float]) -> str:
    if len(x) < 3:
        return "insufficient_pairs"
    if len(set(x)) < 2:
        return "constant_human_scores"
    if len(set(y)) < 2:
        return "constant_asr_scores"
    return "ok"


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def cluster_resamples(rows: list[dict], samples: int, seed: int) -> list[list[int]]:
    """Return reproducible index draws that resample utterance IDs with replacement."""
    clusters: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        clusters[row["utterance_id"]].append(i)
    keys = sorted(clusters)
    if len(keys) < 3:
        return []
    rng = random.Random(seed)
    return [[i for _ in keys for i in clusters[rng.choice(keys)]] for _ in range(samples)]


def association(x: list[float], y: list[float], resamples: list[list[int]] = ()) -> dict:
    status = correlation_status(x, y)
    result = {"n_pairs": len(x), "status": status,
              "pearson_r": pearson(x, y), "spearman_rho": spearman(x, y)}
    estimates: dict[str, list[float]] = {"pearson": [], "spearman": []}
    if status == "ok":
        for indices in resamples:
            a, b = [x[i] for i in indices], [y[i] for i in indices]
            for name, fn in [("pearson", pearson), ("spearman", spearman)]:
                value = fn(a, b)
                if value is not None:
                    estimates[name].append(value)
    for name, values in estimates.items():
        enough = bool(resamples) and len(values) >= max(20, math.ceil(0.8 * len(resamples)))
        result[name + "_ci_low"] = quantile(values, 0.025) if enough else None
        result[name + "_ci_high"] = quantile(values, 0.975) if enough else None
        result[name + "_bootstrap_valid"] = len(values)
    result["bootstrap_requested"] = len(resamples)
    result["ci_status"] = (
        "not_requested_or_too_few_utterance_clusters" if not resamples else
        "ok" if all(len(v) >= max(20, math.ceil(0.8 * len(resamples))) for v in estimates.values()) else
        "too_few_defined_resamples")
    return result


def within_condition_association(rows: list[dict], metric: str,
                                 resamples: list[list[int]] = ()) -> dict:
    """One correlation over all rows after removing each condition's mean.

    Pearson: human and ASR scores are centred on their own condition means
    (the repeated-measures correlation idea, Bakdash & Marusich 2017).
    Spearman: utterances are ranked within each condition and the ranks are
    centred on the condition's mean rank. Differences between conditions
    therefore cannot raise or lower either coefficient. In every bootstrap
    resample the condition means are recomputed before centring.
    """
    def coefficients(sample: list[dict]) -> tuple[float | None, float | None]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in sample:
            groups[row["condition"]].append(row)
        hx, ax, hr, ar = [], [], [], []
        for group in groups.values():
            human = [r[f"human_mean_{metric}"] for r in group]
            asr = [r[f"asr_{metric}"] for r in group]
            hm, am = mean(human), mean(asr)
            hx += [v - hm for v in human]
            ax += [v - am for v in asr]
            human_ranks, asr_ranks = average_ranks(human), average_ranks(asr)
            hrm, arm = mean(human_ranks), mean(asr_ranks)
            hr += [v - hrm for v in human_ranks]
            ar += [v - arm for v in asr_ranks]
        return pearson(hx, ax), pearson(hr, ar)

    r, rho = coefficients(rows)
    result = {
        "n_pairs": len(rows), "n_conditions": len({row["condition"] for row in rows}),
        "status": "ok" if r is not None and rho is not None else
                  "insufficient_pairs" if len(rows) < 3 else "undefined_no_within_condition_variation",
        "pearson_r": r, "spearman_rho": rho,
    }
    estimates: dict[str, list[float]] = {"pearson": [], "spearman": []}
    if result["status"] == "ok":
        for indices in resamples:
            br, brho = coefficients([rows[i] for i in indices])
            if br is not None:
                estimates["pearson"].append(br)
            if brho is not None:
                estimates["spearman"].append(brho)
    needed = max(20, math.ceil(0.8 * len(resamples)))
    for name, values in estimates.items():
        enough = bool(resamples) and len(values) >= needed
        result[name + "_ci_low"] = quantile(values, 0.025) if enough else None
        result[name + "_ci_high"] = quantile(values, 0.975) if enough else None
        result[name + "_bootstrap_valid"] = len(values)
    result["bootstrap_requested"] = len(resamples)
    return result


def write_table(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_comparisons(folder: Path) -> dict[str, list[dict]]:
    """Load canonical score pairs and align every ASR to the same recordings."""
    tables: dict[str, list[dict]] = {}
    canonical_keys = None
    canonical_human = {}
    for path in sorted(folder.glob("comparison_*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"utterance_id", "condition", "audio_file", "speaker_id", "asr_system",
                        "human_n_ratings", "human_mean_wer", "human_mean_cer", "asr_wer", "asr_cer"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{path}: missing fields {sorted(required - set(reader.fieldnames or []))}")
            rows = sorted(list(reader), key=lambda r: (r["utterance_id"], r["condition"]))
        labels = {r["asr_system"] for r in rows}
        if len(labels) != 1:
            raise ValueError(f"{path}: expected one ASR label and at least one row")
        label = next(iter(labels))
        if label in tables:
            raise ValueError(f"Duplicate ASR label: {label}")
        keys = [(r["utterance_id"], r["condition"]) for r in rows]
        if any(not a or not b for a, b in keys) or len(keys) != len(set(keys)):
            raise ValueError(f"{path}: blank/duplicate utterance-condition key")
        for row in rows:
            if int(row["human_n_ratings"]) < 1:
                raise ValueError(f"{path}: recording has no human ratings")
            for metric in ["wer", "cer"]:
                for field in [f"human_mean_{metric}", f"asr_{metric}"]:
                    row[field] = number(row.get(field))
                if row[f"human_mean_{metric}"] is None:
                    raise ValueError(f"{path}: missing human score")

        human_signature = lambda row: tuple(row.get(field) for field in [
            "audio_file", "speaker_id", "gender", "language", "human_n_ratings",
            "human_mean_wer", "human_mean_cer",
        ])
        if canonical_keys is None:
            canonical_keys = keys
            canonical_human = {key: human_signature(row) for key, row in zip(keys, rows)}
        else:
            if keys != canonical_keys:
                raise ValueError(f"{path}: human recording universe differs across comparison files")
            for key, row in zip(keys, rows):
                if human_signature(row) != canonical_human[key]:
                    raise ValueError(f"{path}: inconsistent human/recording metadata for {key}")
        tables[label] = rows
    if not tables:
        raise ValueError(f"No comparison CSVs in {folder}")
    return tables


def _ordered_conditions(rows: list[dict]) -> list[str]:
    preferred = ["mantis_3.2", "mantis_5.2_content_apt", "mantis_v5_w2s_baseline_E2E_apt",
                 "spoken_audio", "whisper_audio"]
    available = {row["condition"] for row in rows}
    return [value for value in preferred if value in available] + sorted(available - set(preferred))


def ranking_summary(rows: list[dict], label: str, metric: str) -> tuple[list[dict], dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    conditions = _ordered_conditions(rows)
    x = [mean(r[f"human_mean_{metric}"] for r in groups[c]) for c in conditions]
    y = [mean(r[f"asr_{metric}"] for r in groups[c]) for c in conditions]
    hx, hy = average_ranks(x), average_ranks(y)
    # Model-level input: one row per condition. Pearson uses the two means, Spearman the two ranks.
    details = [{"asr_system": label, "condition": c, "n_recordings": len(groups[c]),
                f"human_mean_{metric}": x[i], f"asr_mean_{metric}": y[i],
                "human_rank": hx[i], "asr_rank": hy[i]} for i, c in enumerate(conditions)]
    correct = reversed_pairs = asr_ties = human_ties = 0
    tx, ty = [round(v, TIE_DECIMALS) for v in x], [round(v, TIE_DECIMALS) for v in y]
    for i in range(len(x)):
        for j in range(i):
            if tx[i] == tx[j]:
                human_ties += 1
            elif ty[i] == ty[j]:
                asr_ties += 1
            elif (tx[i] > tx[j]) == (ty[i] > ty[j]):
                correct += 1
            else:
                reversed_pairs += 1
    denominator = correct + reversed_pairs + asr_ties
    summary = {"asr_system": label, "metric": metric, "n_conditions": len(conditions),
               "pearson_r": pearson(x, y),
               "spearman_rho": spearman(x, y), "strict_pairs_correct": correct,
               "strict_pairs_reversed": reversed_pairs, "asr_tied_strict_pairs": asr_ties,
               "human_tied_pairs": human_ties,
               "strict_pair_agreement": correct / denominator if denominator else None,
               "interpretation": "descriptive_only_few_conditions"}
    return details, summary


def utterance_pairs(rows: list[dict], label: str, metric: str, anchors: set[str]) -> list[dict]:
    """Utterance-level input: one row per recording, with every column a correlation uses.

    Per-condition Pearson   = pearson(human_score, asr_score) over one condition's rows.
    Per-condition Spearman  = pearson(human_rank, asr_rank) over the same rows.
    Within-condition Pearson  = pearson(human_centred, asr_centred) over all (or generated) rows.
    Within-condition Spearman = pearson(human_rank_centred, asr_rank_centred) over the same rows.
    Ranks are within the condition (1 = lowest error, ties share the average rank);
    centred = value minus that condition's mean.
    """
    out = []
    for condition in _ordered_conditions(rows):
        group = sorted((r for r in rows if r["condition"] == condition), key=lambda r: r["utterance_id"])
        human = [r[f"human_mean_{metric}"] for r in group]
        asr = [r[f"asr_{metric}"] for r in group]
        human_ranks, asr_ranks = average_ranks(human), average_ranks(asr)
        hm, am, hrm, arm = mean(human), mean(asr), mean(human_ranks), mean(asr_ranks)
        for r, h, a, hr, ar in zip(group, human, asr, human_ranks, asr_ranks):
            out.append({
                "asr_system": label, "condition": condition, "is_anchor": condition in anchors,
                "utterance_id": r["utterance_id"],
                f"human_mean_{metric}": h, f"asr_{metric}": a,
                "human_rank": hr, "asr_rank": ar,
                "human_centred": h - hm, "asr_centred": a - am,
                "human_rank_centred": hr - hrm, "asr_rank_centred": ar - arm,
            })
    return out


def analyze(tables: dict[str, list[dict]], output: Path, samples: int = 2000, seed: int = 20261102,
            allow_incomplete: bool = False, anchor_conditions: list[str] | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    anchors = set(anchor_conditions or [])
    labels = sorted(tables)
    expected = len(tables[labels[0]])
    coverage, common = [], {}
    for metric in ["wer", "cer"]:
        common[metric] = [i for i in range(expected) if all(tables[label][i]["asr_" + metric] is not None for label in labels)]
        for label in labels:
            n = sum(r["asr_" + metric] is not None for r in tables[label])
            coverage.append({"asr_system": label, "metric": metric, "n_expected": expected,
                             "n_valid_asr": n, "n_missing_asr": expected - n,
                             "n_common_across_asr": len(common[metric])})
    write_table(output / "coverage.csv", coverage)
    if not allow_incomplete and any(r["n_missing_asr"] for r in coverage):
        raise ValueError("Missing ASR scores: inspect coverage.csv; default analysis requires complete coverage")
    if any(len(indices) < 3 for indices in common.values()):
        raise ValueError("Fewer than three common recording pairs; inspect coverage.csv")
    utterance_level, model_summaries, disagreements = [], [], []
    within_condition = []
    pairs: dict[str, list[dict]] = {"wer": [], "cer": []}
    model_inputs: dict[str, list[dict]] = {"wer": [], "cer": []}
    wer_rows_by_asr, wer_summaries = {}, {}
    for metric in ["wer", "cer"]:
        indices = common[metric]
        template = [tables[labels[0]][i] for i in indices]
        conditions = sorted({row["condition"] for row in template})
        condition_resamples = {
            condition: cluster_resamples(
                [row for row in template if row["condition"] == condition], samples, seed
            )
            for condition in conditions
        }
        for label in labels:
            rows = [tables[label][i] for i in indices]
            for condition in conditions:
                group = [r for r in rows if r["condition"] == condition]
                result = {
                    "asr_system": label, "metric": metric, "condition": condition,
                    **association(
                        [r[f"human_mean_{metric}"] for r in group],
                        [r[f"asr_{metric}"] for r in group],
                        condition_resamples[condition],
                    ),
                }
                utterance_level.append(result)
                if metric == "wer":
                    wer_summaries[label, condition] = result
            pairs[metric].extend(utterance_pairs(rows, label, metric, anchors))
            detail, ranking = ranking_summary(rows, label, metric)
            model_inputs[metric].extend(detail)
            model_summaries.append(ranking)
            if metric == "wer":
                wer_rows_by_asr[label] = rows
        condition_sets = [("all_conditions", set(conditions))]
        generated = set(conditions) - anchors
        if anchors & set(conditions) and generated:
            condition_sets.append(("generated_only", generated))
        for set_name, included in condition_sets:
            positions = [k for k, row in enumerate(template) if row["condition"] in included]
            resamples = cluster_resamples([template[k] for k in positions], samples, seed)
            for label in labels:
                rows = [tables[label][indices[k]] for k in positions]
                within_condition.append({
                    "asr_system": label, "metric": metric, "condition_set": set_name,
                    "conditions": ";".join(c for c in _ordered_conditions(rows)),
                    **within_condition_association(rows, metric, resamples),
                })
        for label in labels:
            rows = [tables[label][i] for i in indices]
            deltas = [round(r[f"asr_{metric}"] - r[f"human_mean_{metric}"], TIE_DECIMALS) for r in rows]
            discrepancy_ranks = average_ranks([-abs(v) for v in deltas])
            for row, delta, rank in sorted(zip(rows, deltas, discrepancy_ranks), key=lambda p: p[2]):
                disagreements.append({**row, "metric": metric, "absolute_disagreement_rank": rank,
                    "delta_asr_minus_human_mean": delta, "absolute_disagreement": abs(delta),
                    "review_category": "", "review_notes": ""})
    utterance_dir, model_dir, disagreement_dir = (output / "utterance_level", output / "model_level",
                                                  output / "disagreement")
    for metric in ["wer", "cer"]:
        write_table(utterance_dir / f"01_pairs_{metric}.csv", pairs[metric])
        write_table(model_dir / f"01_condition_means_{metric}.csv", model_inputs[metric])
    write_table(utterance_dir / "02_correlations_by_condition.csv", utterance_level)
    write_table(utterance_dir / "03_correlations_within_condition.csv", within_condition)
    write_table(model_dir / "02_correlations.csv", model_summaries)
    write_table(disagreement_dir / "disagreements.csv", disagreements)

    conditions = _ordered_conditions(next(iter(wer_rows_by_asr.values())))
    coefficient_fields = ["asr_system", *conditions]
    spearman_table = [{"asr_system": label, **{
        condition: wer_summaries[label, condition]["spearman_rho"] for condition in conditions
    }} for label in labels]
    pearson_table = [{"asr_system": label, **{
        condition: wer_summaries[label, condition]["pearson_r"] for condition in conditions
    }} for label in labels]
    write_table(utterance_dir / "wer_spearman_by_condition.csv", spearman_table, coefficient_fields)
    write_table(utterance_dir / "wer_pearson_by_condition.csv", pearson_table, coefficient_fields)

    disagreement_summary = []
    for label in labels:
        for condition in conditions:
            group = [row for row in wer_rows_by_asr[label] if row["condition"] == condition]
            deltas = [round(row["asr_wer"] - row["human_mean_wer"], TIE_DECIMALS) for row in group]
            disagreement_summary.append({
                "asr_system": label, "condition": condition, "n_pairs": len(group),
                "mean_delta_wer": mean(deltas),
                "mean_absolute_delta_wer": mean(abs(value) for value in deltas),
                "root_mean_squared_delta_wer": math.sqrt(mean(value * value for value in deltas)),
                "n_asr_higher": sum(value > 0 for value in deltas),
                "n_human_higher": sum(value < 0 for value in deltas),
                "n_equal": sum(value == 0 for value in deltas),
            })
    write_table(disagreement_dir / "wer_disagreement_summary.csv", disagreement_summary)

    return {"n_asr_systems": len(labels), "n_expected_recordings": expected,
            "n_common_wer": len(common["wer"]), "n_common_cer": len(common["cer"]),
            "n_utterance_clusters": len({r["utterance_id"] for r in tables[labels[0]]}),
            "n_conditions": len({r["condition"] for r in tables[labels[0]]}),
            "anchor_conditions": sorted(anchors),
            "bootstrap_samples": samples, "random_seed": seed}
