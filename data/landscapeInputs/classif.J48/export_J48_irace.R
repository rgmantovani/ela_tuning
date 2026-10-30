# Exports one CSV per dataset with the full irace optimization history for classif.J48.
# Usage: Rscript export_J48_irace.R

TUNING_BASE <- "/Users/rgmantovani/Documents/Development/tuningData/dt_tuning"
OUT_DIR     <- "/Users/rgmantovani/Documents/Development/ela_tuning/data/landscapeInputs/classif.J48"
ALG         <- "classif.J48"
OPT         <- "irace"
LEARNER_KEY <- "classif.J48.imputed.preproc.tuned"

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

datasets <- list.dirs(TUNING_BASE, full.names = FALSE, recursive = FALSE)

for (ds in datasets) {

  opt_dir <- file.path(TUNING_BASE, ds, ALG, OPT)
  if (!dir.exists(opt_dir)) next

  reps <- list.dirs(opt_dir, full.names = TRUE, recursive = FALSE)
  if (length(reps) == 0) next

  all_rows <- list()

  for (rep_path in reps) {
    rep_num    <- as.integer(gsub("rep", "", basename(rep_path)))
    rdata_file <- file.path(rep_path, paste0("opt_params_", ds, ".RData"))
    if (!file.exists(rdata_file)) next

    e <- new.env()
    suppressWarnings(load(rdata_file, envir = e))
    obj   <- get(ls(e)[1], envir = e)
    inner <- obj[[ds]][[LEARNER_KEY]]
    if (is.null(inner)) next

    for (iter_idx in seq_along(inner)) {
      op <- inner[[iter_idx]][["opt.path"]]
      if (is.null(op)) next
      df <- tryCatch(as.data.frame(op), error = function(e) NULL)
      if (is.null(df) || nrow(df) == 0) next

      df$dataset    <- ds
      df$rep        <- rep_num
      df$tuner_iter <- iter_idx
      all_rows[[length(all_rows) + 1]] <- df
    }
  }

  if (length(all_rows) == 0) {
    cat("No data found for:", ds, "\n")
    next
  }

  combined <- do.call(rbind, all_rows)

  # Reorder: identifiers first, then hyperparameters, then objective + metadata
  hp_cols   <- c("O", "R", "C", "M", "N", "B", "S", "A", "J")
  meta_cols <- c("ber.test.mean", "dob", "eol", "error.message", "exec.time")
  id_cols   <- c("dataset", "rep", "tuner_iter")
  all_cols  <- c(id_cols, hp_cols, meta_cols)
  combined  <- combined[, intersect(all_cols, colnames(combined))]

  out_file <- file.path(OUT_DIR, paste0("J48_", ds, ".csv"))
  write.csv(combined, out_file, row.names = FALSE)
  cat("Saved:", basename(out_file), " (", nrow(combined), "rows )\n")
}

cat("\nDone. Files saved to:", OUT_DIR, "\n")
