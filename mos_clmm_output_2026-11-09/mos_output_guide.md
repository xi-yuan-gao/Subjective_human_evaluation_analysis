# MOS ordinal mixed-effects output guide

How to read the output of `python run_mos_clmm.py`.

## The analysis in one paragraph

MOS ratings (1–5) are compared between two conditions with a cumulative link mixed model
(CLMM; R package `ordinal`), which treats ratings as ordered categories. The model estimates
how much more likely one condition is to receive a higher rating than the other, at any point
on the scale.

```
mos_rating ~ condition + speaker_id
             + (1 | listener_id) + (1 | listener_condition) + (1 | item_id)
```

- **Fixed effects:** condition (the comparison of interest) and speaker (control variable).
- **Random effects:** listener (overall rating level), listener × condition (each listener's
  own preference between conditions), item (each stimulus clip's own quality).
- **Test:** likelihood ratio test (LRT) for the condition effect.
- **Effect size:** cumulative odds ratio, reference vs comparison, with 95% Wald CI.

Run:

```
python run_mos_clmm.py --input <mos_input.csv> --output-root <new_dir> --reference-condition <condition name>
```

Always set `--reference-condition` explicitly; the name must match the `condition` values in
the input exactly. Requirements: R with the packages `ordinal` and `emmeans`.

## Reading order

| Step | File | Question it answers |
|---|---|---|
| 1 | `clmm_summary.txt` | What is the overall result? |
| 2 | `model_diagnostics.csv` | Did the model fit properly? |
| 3 | `condition_overall_lrt.csv` | **Main test:** do the conditions differ? |
| 4 | `e2e_pairwise_wald_tests.csv` | **Main effect size:** by how much, in which direction? |
| 5 | `model_audit.txt` | How much do listeners and items vary? |
| — | `clmm_coefficients.csv` | Same numbers as `model_audit.txt`; skip |
| — | `model_warnings.txt` | Fitting warnings; empty is good |

## Key concepts

**Cumulative odds ratio.** How much more likely the reference condition is to receive a
higher rating than the comparison condition, at any point on the scale (for example 4–5 vs
1–3, or 3–5 vs 1–2).

- 1 = both conditions rated equally.
- Above 1 = the **reference** condition was rated higher.
- Below 1 = the **reference** condition was rated lower.
- Example: 0.72 means the odds of a higher rating for the reference were 0.72 times those of
  the comparison.

**Significance.** A difference is reliable when the 95% CI of the odds ratio excludes 1
(equivalently, LRT p < .05). If the CI includes 1, the data cannot show which condition was
rated higher; this is not evidence that they are equal.

**Size.** Check how close the CI limits are to 1. A limit close to 1 means the difference is
reliable but may be small.

**Typical listener and stimulus.** The odds ratio describes a listener and stimulus with
average random effects, after accounting for speaker.

**Log-odds scale.** Coefficients, CIs before conversion, and random-effect SDs are on the
log-odds scale. The odds ratio is e^(log-odds difference). No effect = 0 on the log-odds
scale = 1 as an odds ratio.

## File reference

### `model_diagnostics.csv`

| Column | Good value | Meaning |
|---|---|---|
| `convergence_code` | 0 | The optimizer finished successfully |
| `maximum_absolute_gradient` | ≤ 0.001 ideal; slightly above is usually fine | How close the fit is to the exact optimum |
| `minimum_hessian_eigenvalue` | > 0 | The estimates are well defined |
| `warnings_captured` | 0 | No warnings from R |
| `status` | PASS | PASS only if all of the above meet the strict thresholds |
| `responses`, `listeners`, `items`, `speakers`, `conditions` | As designed | Check against the experiment |

**REVIEW** with convergence code 0, a positive Hessian eigenvalue, no warnings, and a gradient
only slightly above 0.001 is acceptable; report that the model converged without warnings.
A gradient far above 0.001, a non-zero convergence code, or a non-positive eigenvalue needs
attention before reporting.

### `condition_overall_lrt.csv` — main test

| Column | Meaning |
|---|---|
| `chi_square` | How much knowing the condition improves the fit |
| `degrees_of_freedom` | Number of condition parameters (1 for two conditions) |
| `p_value` | **Report this p-value.** Below .05 = reliable difference |
| `full_model`, `reduced_model` | The two models compared (with and without condition) |

Report as χ²(df) = chi_square, p = p_value.

### `e2e_pairwise_wald_tests.csv` — main effect size

| Column | Meaning |
|---|---|
| `reference_condition`, `comparison_condition` | Direction of the comparison |
| `estimated_log_odds_difference` | Reference minus comparison, log-odds scale |
| `standard_error` | Precision of that difference |
| `z_statistic` | Estimate ÷ SE |
| `cumulative_odds_ratio` | **Report this.** e^(log-odds difference) |
| `odds_ratio_ci_95_lower`, `_upper` | **Report this** 95% CI |
| `p_value` | Wald p-value; should agree with the LRT. Report the LRT p instead |

### `model_audit.txt`

Full printout of both models, the LRT table, random effects and R session information.

**Random effects** (full model, "Random effects" block): one row per group, with variance,
SD and number of groups. Compare the SDs with the size of the condition effect
(`estimated_log_odds_difference`, ignoring the sign):

| Group | What the SD describes |
|---|---|
| `listener_id` | How much listeners differ in overall rating level (harsh vs generous) |
| `listener_condition` | How much listeners differ in their preference between conditions |
| `item_id` | How much stimulus clips differ in quality |

An SD larger than the condition effect means that source varies more than the conditions
differ. Number of groups for `listener_condition` should be listeners × conditions (every
listener rated every condition).

**Coefficients** (full model): the condition coefficient here is **comparison minus
reference** (the sign is opposite to `e2e_pairwise_wald_tests.csv`). It is the same result;
report the pairwise file. Speaker coefficients compare each speaker with the reference speaker
(the first alphabetically); they are a control and not part of the condition comparison.

**Threshold coefficients** (1|2, 2|3, 3|4, 4|5): the points on the latent scale where ratings
switch to the next category. Roughly even spacing means listeners used the scale evenly.
Usually not reported.

**Session information:** R and package versions, for the methods section.

## Checklist before reporting

- [ ] Reference condition is the intended one (check `reference_condition`).
- [ ] Sample sizes in `model_diagnostics.csv` match the design.
- [ ] Convergence code 0, positive Hessian eigenvalue, no warnings.
- [ ] Every listener rated every condition (`listener_condition` groups = listeners × conditions).
- [ ] LRT and Wald p-values agree.
- [ ] Odds ratio direction stated in words (which condition was rated higher).

## Reporting template

**Methods**

> MOS ratings were analyzed with a cumulative link mixed model (logit link, flexible
> thresholds; R package ordinal [version], R [version]), which treats ratings as ordered
> categories. Condition and speaker were fixed effects. Random intercepts were included for
> listeners, for each listener × condition combination (allowing listeners to differ in their
> preference between conditions), and for stimulus items. The condition effect was tested with
> a likelihood ratio test, and its size is expressed as a cumulative odds ratio
> ([reference] vs [comparison]) with a 95% Wald confidence interval. The analysis included
> [responses] ratings from [listeners] listeners on [items] stimuli from [speakers] speakers.

**Results**

> The two conditions [differed / did not differ reliably] in MOS (likelihood ratio test,
> χ²(1) = [chi_square], p = [p_value]). The cumulative odds ratio ([reference] vs
> [comparison]) was [odds ratio] (95% CI [lower] to [upper]): for a typical listener and
> stimulus, the odds of [reference] receiving a higher rating were [odds ratio] times those of
> [comparison]. [State in words which condition was rated higher, and whether the CI limit
> close to 1 suggests the difference may be modest.] Listeners varied in their overall rating
> level (SD [listener SD] on the log-odds scale) and in their preference between conditions
> (SD [listener × condition SD]). The model converged without warnings.

**How to read the results (for general readers)**

> We compare the conditions with a cumulative odds ratio: how much more likely one condition
> is to receive a higher rating than the other, at any point on the 1–5 scale. An odds ratio
> of 1 means both were rated equally; above 1, the reference condition tended to receive
> higher ratings; below 1, lower ratings. The 95% confidence interval is the range of values
> consistent with the data; if it includes 1, the data cannot show which condition was rated
> higher. The likelihood ratio test compares the statistical model with and without
> information about the condition; p below .05 indicates a reliable difference.

## Note on more than two conditions

The script compares the reference with every other condition but applies no
multiple-comparison correction. With three or more conditions, add a correction (for example
Holm) before reporting pairwise p-values.
