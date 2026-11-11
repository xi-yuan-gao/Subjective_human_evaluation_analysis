# Count-based generalized linear mixed models for transcription errors.
#
# Called by mixed_effects.py. Do not run by hand unless debugging:
#   Rscript glmm_count_models.R <model_input.csv> <output_dir> <seed> <n_simulations>
#
# Input columns: listener_id, utterance_id, model_c,
#                word_errors, reference_words, character_errors, reference_characters
# model_c is +0.5 for the first selected condition and -0.5 for the second.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: Rscript glmm_count_models.R <input.csv> <output_dir> <seed> <n_simulations>")
}
input_path <- args[1]
output_dir <- args[2]
seed <- as.integer(args[3])
n_sim <- as.integer(args[4])

if (!requireNamespace("glmmTMB", quietly = TRUE)) {
  stop("R package 'glmmTMB' is not installed. Install it in R with: install.packages('glmmTMB')")
}
suppressPackageStartupMessages(library(glmmTMB))

d <- read.csv(input_path, stringsAsFactors = FALSE)
d$listener_id <- factor(d$listener_id)
d$utterance_id <- factor(d$utterance_id)

outcomes <- list(
  WER = c(errors = "word_errors", denominator = "reference_words"),
  CER = c(errors = "character_errors", denominator = "reference_characters")
)

# Utterance random-effects structures, tried in this pre-specified order.
utterance_structures <- c(
  full = "(1 + model_c | utterance_id)",
  diag = "diag(1 + model_c | utterance_id)",
  intercept = "(1 | utterance_id)"
)
listener_structure <- "(1 | listener_id)"

SD_FLOOR <- 1e-3   # random-effect SD below this (log scale) is treated as singular
CORR_CEILING <- 0.99

fit_quietly <- function(formula, data, family) {
  warnings_seen <- character(0)
  model <- tryCatch(
    withCallingHandlers(
      glmmTMB(formula, data = data, family = family),
      warning = function(w) {
        warnings_seen <<- c(warnings_seen, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      warnings_seen <<- c(warnings_seen, paste("ERROR:", conditionMessage(e)))
      NULL
    }
  )
  list(model = model, warnings = unique(warnings_seen))
}

random_effect_table <- function(model) {
  vc <- VarCorr(model)$cond
  rows <- list()
  for (group in names(vc)) {
    sds <- attr(vc[[group]], "stddev")
    corr <- attr(vc[[group]], "correlation")
    for (term in names(sds)) {
      rows[[length(rows) + 1]] <- data.frame(
        group = group, term = term, type = "sd", value = unname(sds[term])
      )
    }
    if (!is.null(corr) && length(sds) > 1) {
      terms <- names(sds)
      for (i in seq_along(terms)) for (j in seq_along(terms)) if (i < j) {
        rows[[length(rows) + 1]] <- data.frame(
          group = group, term = paste(terms[i], terms[j], sep = " ~ "),
          type = "correlation", value = corr[i, j]
        )
      }
    }
  }
  do.call(rbind, rows)
}

assess_fit <- function(fit) {
  model <- fit$model
  if (is.null(model)) {
    return(list(ok = FALSE, reason = "model failed to fit"))
  }
  if (!isTRUE(model$fit$convergence == 0)) {
    return(list(ok = FALSE, reason = paste("optimizer convergence code", model$fit$convergence)))
  }
  if (!isTRUE(model$sdr$pdHess)) {
    return(list(ok = FALSE, reason = "non-positive-definite Hessian"))
  }
  se <- sqrt(diag(vcov(model)$cond))
  if (any(!is.finite(se))) {
    return(list(ok = FALSE, reason = "non-finite fixed-effect standard errors"))
  }
  re <- random_effect_table(model)
  # Only the utterance structure is simplified. The listener intercept is
  # required by the design and is kept even if its variance is near zero.
  utt <- re$group == "utterance_id"
  small_sd <- utt & re$type == "sd" & re$value < SD_FLOOR
  if (any(small_sd)) {
    return(list(ok = FALSE, reason = paste(
      "singular: utterance random-effect SD near zero for",
      paste(re$term[small_sd], collapse = "; ")
    )))
  }
  boundary_corr <- utt & re$type == "correlation" & abs(re$value) > CORR_CEILING
  if (any(boundary_corr)) {
    return(list(ok = FALSE, reason = "singular: utterance random-effect correlation at +/-1"))
  }
  listener_sd <- re$value[re$group == "listener_id" & re$type == "sd"]
  if (length(listener_sd) && listener_sd < SD_FLOOR) {
    return(list(ok = TRUE, reason = "accepted; listener variance near zero (boundary), retained by design"))
  }
  list(ok = TRUE, reason = "accepted")
}

build_formula <- function(utterance_term, include_condition = TRUE) {
  fixed <- if (include_condition) "y ~ model_c" else "y ~ 1"
  as.formula(paste(fixed, "+ offset(log(denominator)) +", listener_structure, "+", utterance_term))
}

summarise_model <- function(model, outcome, role, family, structure_name, null_model) {
  b <- fixef(model)$cond
  v <- vcov(model)$cond
  lrt_chisq <- NA_real_
  lrt_p <- NA_real_
  if (!is.null(null_model)) {
    lrt_chisq <- as.numeric(2 * (logLik(model) - logLik(null_model)))
    if (is.finite(lrt_chisq) && lrt_chisq < 0 && lrt_chisq > -1e-6) lrt_chisq <- 0
    lrt_p <- pchisq(lrt_chisq, df = 1, lower.tail = FALSE)
  }
  data.frame(
    outcome = outcome,
    role = role,
    family = family,
    utterance_random_effects = structure_name,
    utterance_formula_term = utterance_structures[[structure_name]],
    n_obs = nobs(model),
    log_likelihood = as.numeric(logLik(model)),
    aic = AIC(model),
    intercept = unname(b["(Intercept)"]),
    condition_coef = unname(b["model_c"]),
    var_intercept = v["(Intercept)", "(Intercept)"],
    var_condition = v["model_c", "model_c"],
    cov_intercept_condition = v["(Intercept)", "model_c"],
    nb_theta = if (family == "nbinom2") sigma(model) else NA_real_,
    lrt_chi_square = lrt_chisq,
    lrt_df = 1,
    lrt_p_value = lrt_p,
    null_model_converged = !is.null(null_model),
    stringsAsFactors = FALSE
  )
}

simulate_check <- function(model, y, family, outcome, role) {
  # Whole-model simulation (posterior-predictive style): simulate complete new
  # datasets from the fitted model, including new random effects, and compare
  # features of the observed counts with their simulated distributions.
  # Unlike checks based on fitted values, this is not biased by overfitting.
  sims <- simulate(model, nsim = n_sim, seed = seed)
  sim_zero <- vapply(sims, function(s) sum(s == 0), numeric(1))
  sim_var <- vapply(sims, function(s) var(s), numeric(1))
  sim_max <- vapply(sims, function(s) max(s), numeric(1))
  two_sided <- function(obs, values) {
    min(1, 2 * min(mean(values >= obs), mean(values <= obs)))
  }
  data.frame(
    outcome = outcome,
    role = role,
    family = family,
    n_simulations = n_sim,
    observed_zero_count = sum(y == 0),
    mean_simulated_zero_count = mean(sim_zero),
    zero_ratio = sum(y == 0) / mean(sim_zero),
    zero_p_value = two_sided(sum(y == 0), sim_zero),
    observed_count_variance = var(y),
    mean_simulated_count_variance = mean(sim_var),
    variance_ratio = var(y) / mean(sim_var),
    variance_p_value = two_sided(var(y), sim_var),
    observed_max_count = max(y),
    mean_simulated_max_count = mean(sim_max),
    max_p_value = two_sided(max(y), sim_max),
    stringsAsFactors = FALSE
  )
}

attempt_rows <- list()
result_rows <- list()
random_rows <- list()
diagnostic_rows <- list()

for (outcome in names(outcomes)) {
  cols <- outcomes[[outcome]]
  d$y <- d[[cols[["errors"]]]]
  d$denominator <- d[[cols[["denominator"]]]]

  # 1. Primary model: negative binomial, simplest acceptable structure in order.
  primary <- NULL
  primary_structure <- NA_character_
  for (structure_name in names(utterance_structures)) {
    fit <- fit_quietly(build_formula(utterance_structures[[structure_name]]), d, "nbinom2")
    verdict <- assess_fit(fit)
    attempt_rows[[length(attempt_rows) + 1]] <- data.frame(
      outcome = outcome, family = "nbinom2", utterance_random_effects = structure_name,
      accepted = verdict$ok, reason = verdict$reason,
      warnings = paste(fit$warnings, collapse = " | "), stringsAsFactors = FALSE
    )
    if (verdict$ok) {
      primary <- fit$model
      primary_structure <- structure_name
      break
    }
  }
  if (is.null(primary)) {
    stop(paste("No acceptable negative binomial model for", outcome,
               "- see model_selection_attempts.csv"))
  }

  fit_role <- function(role, family, structure_name) {
    fit <- fit_quietly(build_formula(utterance_structures[[structure_name]]), d, family)
    verdict <- assess_fit(fit)
    if (is.null(fit$model)) {
      return(list(row = NULL, model = NULL, verdict = verdict, warnings = fit$warnings))
    }
    null_fit <- fit_quietly(build_formula(utterance_structures[[structure_name]], FALSE), d, family)
    null_ok <- !is.null(null_fit$model) && isTRUE(null_fit$model$fit$convergence == 0)
    row <- summarise_model(fit$model, outcome, role, family, structure_name,
                           if (null_ok) null_fit$model else NULL)
    row$fit_accepted <- verdict$ok
    row$fit_note <- verdict$reason
    row$warnings <- paste(unique(c(fit$warnings, null_fit$warnings)), collapse = " | ")
    list(row = row, model = fit$model, verdict = verdict)
  }

  primary_result <- fit_role("primary", "nbinom2", primary_structure)
  result_rows[[length(result_rows) + 1]] <- primary_result$row
  re <- random_effect_table(primary)
  re$outcome <- outcome
  re$role <- "primary"
  random_rows[[length(random_rows) + 1]] <- re

  # 2. Sensitivity: Poisson with the same random-effects structure.
  poisson_result <- fit_role("sensitivity_poisson", "poisson", primary_structure)
  if (!is.null(poisson_result$row)) {
    result_rows[[length(result_rows) + 1]] <- poisson_result$row
  }

  # 3. Sensitivity: negative binomial with utterance random intercepts only.
  if (primary_structure != "intercept") {
    intercept_result <- fit_role("sensitivity_intercept_only", "nbinom2", "intercept")
    if (!is.null(intercept_result$row)) {
      result_rows[[length(result_rows) + 1]] <- intercept_result$row
    }
  }

  # 4. Simulation-based distribution checks.
  diagnostic_rows[[length(diagnostic_rows) + 1]] <-
    simulate_check(primary, d$y, "nbinom2", outcome, "primary")
  if (!is.null(poisson_result$model)) {
    diagnostic_rows[[length(diagnostic_rows) + 1]] <-
      simulate_check(poisson_result$model, d$y, "poisson", outcome, "sensitivity_poisson")
  }
}

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
write.csv(do.call(rbind, attempt_rows), file.path(output_dir, "attempts.csv"), row.names = FALSE)
write.csv(do.call(rbind, result_rows), file.path(output_dir, "results.csv"), row.names = FALSE)
write.csv(do.call(rbind, random_rows), file.path(output_dir, "random_effects.csv"), row.names = FALSE)
write.csv(do.call(rbind, diagnostic_rows), file.path(output_dir, "diagnostics.csv"), row.names = FALSE)
writeLines(
  c(
    R.version.string,
    paste("glmmTMB", as.character(packageVersion("glmmTMB"))),
    paste("TMB", as.character(packageVersion("TMB")))
  ),
  file.path(output_dir, "versions.txt")
)
