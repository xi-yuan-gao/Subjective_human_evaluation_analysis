#!/usr/bin/env Rscript

# Frequentist cumulative-link mixed model for the cleaned MOS dataset.
#
# Primary model:
#   mos_rating ~ condition + speaker_id + (1 | listener_id) + (1 | item_id)
#
# Outputs an overall likelihood-ratio test for condition and focused Wald
# z-tests comparing the configured reference condition with every other
# condition. Pairwise p-values are corrected together using Holm's method.

parse_arguments <- function(arguments) {
  values <- list()
  index <- 1
  while (index <= length(arguments)) {
    key <- arguments[[index]]
    if (!startsWith(key, "--") || index == length(arguments)) {
      stop("Arguments must be supplied as --name value pairs.", call. = FALSE)
    }
    values[[substring(key, 3)]] <- arguments[[index + 1]]
    index <- index + 2
  }
  values
}

required_argument <- function(arguments, name) {
  value <- arguments[[name]]
  if (is.null(value) || !nzchar(value)) {
    stop(sprintf("Missing required argument: --%s", name), call. = FALSE)
  }
  value
}

required_packages <- c("ordinal", "emmeans")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Missing R packages: ", paste(missing_packages, collapse = ", "),
      ". Install them with: install.packages(c(\"ordinal\", \"emmeans\"))"
    ),
    call. = FALSE
  )
}

arguments <- parse_arguments(commandArgs(trailingOnly = TRUE))
input_path <- required_argument(arguments, "input")
output_root <- required_argument(arguments, "output-root")
reference_condition <- required_argument(arguments, "reference-condition")

if (!file.exists(input_path)) {
  stop(sprintf("Input file does not exist: %s", input_path), call. = FALSE)
}
if (dir.exists(output_root)) {
  existing <- list.files(output_root, all.files = TRUE, no.. = TRUE)
  if (length(existing) > 0) {
    stop(sprintf("Output directory is not empty: %s", output_root), call. = FALSE)
  }
} else {
  dir.create(output_root, recursive = TRUE)
}

data <- read.csv(
  input_path,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  na.strings = c("", "NA")
)

required_columns <- c(
  "participant_id", "session_id", "item_id", "condition",
  "speaker_id", "mos_rating"
)
missing_columns <- setdiff(required_columns, names(data))
if (length(missing_columns) > 0) {
  stop(
    paste("Input is missing required columns:", paste(missing_columns, collapse = ", ")),
    call. = FALSE
  )
}
if (nrow(data) == 0) {
  stop("MOS input contains no rows.", call. = FALSE)
}

identifier_columns <- c(
  "participant_id", "session_id", "item_id", "condition", "speaker_id"
)
for (column in identifier_columns) {
  values <- trimws(as.character(data[[column]]))
  if (any(is.na(values) | values == "")) {
    stop(sprintf("Input contains missing %s values.", column), call. = FALSE)
  }
  data[[column]] <- values
}

ratings <- suppressWarnings(as.numeric(data$mos_rating))
if (any(is.na(ratings)) || any(ratings != floor(ratings)) || any(!ratings %in% 1:5)) {
  stop("MOS ratings must be whole-number values from 1 through 5.", call. = FALSE)
}

data$mos_rating_ordered <- ordered(ratings, levels = 1:5)
data$listener_id <- interaction(
  data$participant_id, data$session_id, drop = TRUE, sep = "::"
)
data$item_id <- factor(data$item_id)
data$speaker_id <- factor(data$speaker_id)
data$condition <- factor(data$condition)
if (!reference_condition %in% levels(data$condition)) {
  stop(
    sprintf("Reference condition is not present in the data: %s", reference_condition),
    call. = FALSE
  )
}
data$condition <- stats::relevel(data$condition, ref = reference_condition)

captured_warnings <- character()
capture_fit_warnings <- function(expression) {
  withCallingHandlers(
    expression,
    warning = function(warning) {
      captured_warnings <<- c(captured_warnings, conditionMessage(warning))
      invokeRestart("muffleWarning")
    }
  )
}

control <- ordinal::clmm.control(maxIter = 200, gradTol = 1e-5)
full_model <- capture_fit_warnings(
  ordinal::clmm(
    mos_rating_ordered ~ condition + speaker_id +
      (1 | listener_id) + (1 | item_id),
    data = data,
    link = "logit",
    threshold = "flexible",
    Hess = TRUE,
    nAGQ = 1,
    control = control
  )
)
reduced_model <- capture_fit_warnings(
  ordinal::clmm(
    mos_rating_ordered ~ speaker_id +
      (1 | listener_id) + (1 | item_id),
    data = data,
    link = "logit",
    threshold = "flexible",
    Hess = TRUE,
    nAGQ = 1,
    control = control
  )
)

# Overall condition test: nested-model likelihood-ratio test.
lrt_table <- as.data.frame(stats::anova(reduced_model, full_model))
lrt_row <- nrow(lrt_table)
find_column <- function(table, candidates) {
  match <- candidates[candidates %in% names(table)]
  if (length(match) == 0) NA_character_ else match[[1]]
}
lrt_column <- find_column(lrt_table, c("LR.stat", "LRT", "Chisq"))
df_column <- find_column(lrt_table, c("df", "Chi Df"))
p_column <- find_column(lrt_table, c("Pr(>Chisq)", "Pr(>Chi)", "p.value"))
if (any(is.na(c(lrt_column, df_column, p_column)))) {
  stop(
    paste("Unexpected likelihood-ratio table columns:", paste(names(lrt_table), collapse = ", ")),
    call. = FALSE
  )
}
overall_test <- data.frame(
  test = "Likelihood-ratio test of the overall condition effect",
  full_model = "condition + speaker + crossed listener/item random intercepts",
  reduced_model = "speaker + crossed listener/item random intercepts",
  chi_square = as.numeric(lrt_table[[lrt_column]][lrt_row]),
  degrees_of_freedom = as.numeric(lrt_table[[df_column]][lrt_row]),
  p_value = as.numeric(lrt_table[[p_column]][lrt_row]),
  stringsAsFactors = FALSE
)
write.csv(
  overall_test,
  file.path(output_root, "condition_overall_lrt.csv"),
  row.names = FALSE
)

# Focused e2e-versus-other contrasts on the latent cumulative-logit scale.
condition_levels <- levels(data$condition)
other_conditions <- setdiff(condition_levels, reference_condition)
contrast_methods <- lapply(other_conditions, function(other_condition) {
  weights <- rep(0, length(condition_levels))
  names(weights) <- condition_levels
  weights[[reference_condition]] <- 1
  weights[[other_condition]] <- -1
  weights
})
names(contrast_methods) <- paste(reference_condition, "vs", other_conditions)

estimated_conditions <- emmeans::emmeans(
  full_model, specs = ~ condition, mode = "latent"
)
focused_contrasts <- emmeans::contrast(
  estimated_conditions, method = contrast_methods
)
raw_tests <- as.data.frame(
  summary(focused_contrasts, infer = c(TRUE, TRUE), level = 0.95, adjust = "none")
)
holm_tests <- as.data.frame(
  summary(focused_contrasts, infer = c(TRUE, TRUE), level = 0.95, adjust = "holm")
)

lower_column <- find_column(raw_tests, c("asymp.LCL", "lower.CL"))
upper_column <- find_column(raw_tests, c("asymp.UCL", "upper.CL"))
z_column <- find_column(raw_tests, c("z.ratio", "t.ratio"))
if (any(is.na(c(lower_column, upper_column, z_column)))) {
  stop(
    paste("Unexpected pairwise table columns:", paste(names(raw_tests), collapse = ", ")),
    call. = FALSE
  )
}
pairwise_tests <- data.frame(
  reference_condition = reference_condition,
  comparison_condition = other_conditions,
  contrast_direction = "reference condition minus comparison condition",
  estimated_log_odds_difference = raw_tests$estimate,
  standard_error = raw_tests$SE,
  z_statistic = raw_tests[[z_column]],
  cumulative_odds_ratio = exp(raw_tests$estimate),
  odds_ratio_ci_95_lower = exp(raw_tests[[lower_column]]),
  odds_ratio_ci_95_upper = exp(raw_tests[[upper_column]]),
  p_value_raw = raw_tests$p.value,
  p_value_holm = holm_tests$p.value,
  test = "Wald z-test on the cumulative-logit scale",
  stringsAsFactors = FALSE
)
write.csv(
  pairwise_tests,
  file.path(output_root, "e2e_pairwise_wald_tests.csv"),
  row.names = FALSE
)

# Coefficients include the four rating thresholds and fixed effects.
coefficient_matrix <- coef(summary(full_model))
coefficient_table <- data.frame(
  parameter = rownames(coefficient_matrix),
  coefficient_matrix,
  row.names = NULL,
  check.names = FALSE
)
coefficient_table$parameter_type <- ifelse(
  coefficient_table$parameter %in% names(full_model$alpha),
  "rating threshold",
  "fixed effect"
)
write.csv(
  coefficient_table,
  file.path(output_root, "clmm_coefficients.csv"),
  row.names = FALSE
)

safe_max_gradient <- function(model) {
  if (is.null(model$gradient)) return(NA_real_)
  max(abs(model$gradient), na.rm = TRUE)
}
safe_min_hessian_eigenvalue <- function(model) {
  if (is.null(model$Hessian)) return(NA_real_)
  min(eigen(model$Hessian, symmetric = TRUE, only.values = TRUE)$values)
}
safe_convergence_code <- function(model) {
  if (is.null(model$optRes$convergence)) return(NA_integer_)
  as.integer(model$optRes$convergence)
}

convergence_code <- safe_convergence_code(full_model)
maximum_gradient <- safe_max_gradient(full_model)
minimum_hessian_eigenvalue <- safe_min_hessian_eigenvalue(full_model)
diagnostic_pass <- (
  !is.na(convergence_code) && convergence_code == 0 &&
    !is.na(maximum_gradient) && maximum_gradient <= 0.001 &&
    !is.na(minimum_hessian_eigenvalue) && minimum_hessian_eigenvalue > 0
)
diagnostics <- data.frame(
  status = ifelse(diagnostic_pass, "PASS", "REVIEW"),
  convergence_code = convergence_code,
  maximum_absolute_gradient = maximum_gradient,
  minimum_hessian_eigenvalue = minimum_hessian_eigenvalue,
  responses = nrow(data),
  listeners = nlevels(data$listener_id),
  items = nlevels(data$item_id),
  speakers = nlevels(data$speaker_id),
  conditions = nlevels(data$condition),
  warnings_captured = length(unique(captured_warnings)),
  stringsAsFactors = FALSE
)
write.csv(
  diagnostics,
  file.path(output_root, "model_diagnostics.csv"),
  row.names = FALSE
)
writeLines(
  unique(captured_warnings),
  file.path(output_root, "model_warnings.txt")
)

summary_lines <- c(
  "Frequentist cumulative-link mixed-effects MOS analysis",
  "",
  paste0(
    "Model: MOS rating ~ condition + speaker + (1|listener) + (1|item); ",
    "logit link and flexible thresholds."
  ),
  sprintf(
    "Overall condition effect: likelihood-ratio chi-square(%d) = %.3f, p = %.6g.",
    overall_test$degrees_of_freedom,
    overall_test$chi_square,
    overall_test$p_value
  ),
  sprintf(
    "Model diagnostic status: %s (convergence code %s; maximum absolute gradient %.6g).",
    diagnostics$status,
    diagnostics$convergence_code,
    diagnostics$maximum_absolute_gradient
  ),
  "",
  apply(pairwise_tests, 1, function(row) {
    sprintf(
      "%s versus %s: cumulative odds ratio = %.3f (95%% CI %.3f to %.3f), z = %.3f, Holm-adjusted p = %.6g.",
      row[["reference_condition"]],
      row[["comparison_condition"]],
      as.numeric(row[["cumulative_odds_ratio"]]),
      as.numeric(row[["odds_ratio_ci_95_lower"]]),
      as.numeric(row[["odds_ratio_ci_95_upper"]]),
      as.numeric(row[["z_statistic"]]),
      as.numeric(row[["p_value_holm"]])
    )
  }),
  "",
  "Positive log-odds differences and odds ratios above 1 favor the reference condition."
)
writeLines(summary_lines, file.path(output_root, "clmm_summary.txt"))

capture.output(
  {
    cat("FULL MODEL\n")
    print(summary(full_model))
    cat("\nREDUCED MODEL\n")
    print(summary(reduced_model))
    cat("\nLIKELIHOOD-RATIO TABLE\n")
    print(lrt_table)
    cat("\nRANDOM EFFECT VARIANCES\n")
    print(ordinal::VarCorr(full_model))
    cat("\nSESSION INFORMATION\n")
    print(sessionInfo())
  },
  file = file.path(output_root, "model_audit.txt")
)

cat(sprintf("CLMM analysis complete. Outputs: %s\n", output_root))
cat(sprintf("Diagnostic status: %s\n", diagnostics$status))

