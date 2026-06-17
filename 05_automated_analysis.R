library(ggplot2)
library(reshape2)

OUT <- "./plots/analysis"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ── Load and aggregate results ────────────────────────────────────────────────
files <- list.files("results", pattern = "loo_metrics_seed_", full.names = TRUE)
all_data <- do.call(rbind, lapply(files, read.csv))

colnames(all_data) <- c("dataset", "algorithm", "threshold", "NumFeatures", "F1", "BAC", "AUC")

all_data$threshold <- factor(all_data$threshold)

alg_order <- c("DecisionTree", "KNN", "LogisticRegression", "NaiveBayes",
                "RandomForest", "SVM_Linear", "SVM_RBF", "XGBoost")
alg_labels <- c("DT", "KNN", "LR", "NB", "RF", "SVM-Lin", "SVM-RBF", "XGBoost")

all_data$algorithm <- factor(all_data$algorithm, levels = alg_order, labels = alg_labels)
all_data$dataset   <- factor(all_data$dataset,
                              levels = c("Baseline", "Ela_metadataset", "Combined"),
                              labels = c("Baseline", "ELA Meta-dataset", "Combined"))

theme_pub <- theme_bw(base_size = 11) +
  theme(strip.background = element_rect(fill = "grey92"),
        legend.position  = "bottom",
        plot.title       = element_text(face = "bold", size = 11))

# ── 1. Violin + boxplot: AUC by dataset ──────────────────────────────────────
g1 <- ggplot(all_data, aes(x = dataset, y = AUC, fill = dataset)) +
  geom_violin(alpha = 0.4, trim = FALSE) +
  geom_boxplot(width = 0.15, outlier.size = 0.8) +
  scale_fill_manual(values = c("Baseline" = "#4e79a7", "ELA Meta-dataset" = "#f28e2b", "Combined" = "#59a14f")) +
  labs(x = NULL, y = "AUC") +
  theme_bw(base_size = 13) +
  theme(axis.text  = element_text(size = 11),
        axis.title = element_text(size = 12),
        legend.position = "none")

ggsave(file.path(OUT, "1_auc_by_dataset.pdf"), g1, width = 5, height = 4)
cat("1_auc_by_dataset.pdf saved\n")

# ── 2. Paired dot plot: ELA vs Baseline per algorithm × threshold ─────────────
agg <- aggregate(AUC ~ algorithm + threshold + dataset, data = all_data, FUN = mean)
agg_wide <- reshape(agg, idvar = c("algorithm", "threshold"),
                    timevar = "dataset", direction = "wide")
colnames(agg_wide)[colnames(agg_wide) == "AUC.Baseline"]         <- "baseline"
colnames(agg_wide)[colnames(agg_wide) == "AUC.ELA Meta-dataset"] <- "ela"
colnames(agg_wide)[colnames(agg_wide) == "AUC.Combined"]         <- "combined"

alg_shapes  <- c(DT = 15, KNN = 16, LR = 17, NB = 18, RF = 8, "SVM-Lin" = 10, "SVM-RBF" = 11, XGBoost = 12)
alg_colours <- RColorBrewer::brewer.pal(8, "Dark2")
names(alg_colours) <- levels(agg_wide$algorithm)

# Stack into long format for faceting: Baseline as x, ELA and Combined as y
agg_long <- rbind(
  data.frame(agg_wide[, c("algorithm", "threshold", "baseline", "ela")],
             comparison = "ELA Meta-dataset vs Baseline",
             y = agg_wide$ela),
  data.frame(agg_wide[, c("algorithm", "threshold", "baseline", "ela")],
             comparison = "Combined vs Baseline",
             y = agg_wide$combined)
)
agg_long$x <- agg_long$baseline

g2 <- ggplot(agg_long, aes(x = x, y = y, colour = algorithm, shape = algorithm)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey60", linewidth = 0.6) +
  geom_point(size = 2.5, alpha = 0.85) +
  scale_colour_manual(values = alg_colours, name = "Algorithm") +
  scale_shape_manual(values  = alg_shapes,  name = "Algorithm") +
  guides(colour = guide_legend(ncol = 1, override.aes = list(size = 3)),
         shape  = guide_legend(ncol = 1)) +
  facet_wrap(~comparison) +
  labs(x = "Baseline AUC", y = "AUC") +
  coord_equal(xlim = c(0.50, 0.85), ylim = c(0.50, 0.85)) +
  theme_bw(base_size = 13) +
  theme(strip.background  = element_rect(fill = "grey92"),
        strip.text        = element_text(size = 12),
        axis.text         = element_text(size = 11),
        axis.title        = element_text(size = 12),
        legend.position   = "right",
        legend.key.size   = unit(0.5, "cm"),
        legend.text       = element_text(size = 10),
        legend.title      = element_text(size = 11, face = "bold"),
        legend.box.background = element_blank())

ggsave(file.path(OUT, "2_paired_auc.pdf"), g2, width = 7, height = 3.5)
cat("2_paired_auc.pdf saved\n")

# ── 4. Heatmap: algorithm × threshold, faceted by dataset ────────────────────
agg3 <- aggregate(AUC ~ algorithm + threshold + dataset, data = all_data, FUN = mean)

best_alg <- aggregate(AUC ~ algorithm + dataset, data = agg3, FUN = mean)
best_alg <- do.call(rbind, lapply(split(best_alg, best_alg$dataset), function(d) {
  d[which.max(d$AUC), ]
}))
agg3$is_best <- paste(agg3$algorithm, agg3$dataset) %in%
                paste(best_alg$algorithm, best_alg$dataset)

g4 <- ggplot(agg3, aes(x = threshold, y = algorithm, fill = AUC)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%.3f", AUC),
                fontface = ifelse(is_best, "bold", "plain")), size = 2.8) +
  scale_fill_gradient2(low = "#d73027", mid = "white", high = "#2166ac",
                       midpoint = 0.75, limits = c(0.50, 0.85)) +
  facet_wrap(~dataset) +
  labs(x = expression(tau), y = NULL, fill = "AUC") +
  theme_pub + theme(legend.position = "right")

ggsave(file.path(OUT, "4_heatmap_alg_threshold.pdf"), g4, width = 9, height = 4)
cat("4_heatmap_alg_threshold.pdf saved\n")

# ── 10–11. SHAP plots (requires 04_shap_analysis.py to be run first) ──────────
library(patchwork)

make_shap_plots <- function(tag, combined_file) {
  sv_file <- file.path("tmp", paste0("shap_values_", tag, ".csv"))
  sx_file <- file.path("tmp", paste0("shap_X_",      tag, ".csv"))

  if (!file.exists(sv_file) || !file.exists(sx_file)) {
    cat("SHAP files not found for", tag, "— run 04_shap_analysis.py first.\n")
    return(invisible(NULL))
  }

  shap <- read.csv(sv_file)
  X    <- read.csv(sx_file)

  mean_shap   <- sort(colMeans(abs(shap)), decreasing = FALSE)
  top10_names <- tail(names(mean_shap), 10)
  feat_levels <- top10_names  # shared y-axis order: least to most important

  # ── beeswarm ──────────────────────────────────────────────────────────────
  shap_long <- data.frame()
  for (f in feat_levels) {
    shap_long <- rbind(shap_long, data.frame(
      feature  = f,
      shap_val = shap[[f]],
      feat_val = scale(X[[f]])[, 1]
    ))
  }
  shap_long$feature <- factor(shap_long$feature, levels = feat_levels)

  g_bee <- ggplot(shap_long, aes(x = shap_val, y = feature, colour = feat_val)) +
    geom_jitter(height = 0.22, size = 1.8, alpha = 0.7) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
    scale_colour_gradient2(low = "#d53e4f", mid = "#ffffbf", high = "#3288bd",
                           midpoint = 0, name = "Feature\nvalue\n(scaled)") +
    labs(x = "SHAP value", y = NULL) +
    theme_bw(base_size = 13) +
    theme(strip.background = element_rect(fill = "grey92"),
          axis.text.y      = element_text(family = "mono", size = 11),
          axis.text.x      = element_text(size = 11),
          axis.title.x     = element_text(size = 12),
          legend.text      = element_text(size = 10),
          legend.title     = element_text(size = 11),
          legend.position  = "right")

  # ── bar ───────────────────────────────────────────────────────────────────
  bar_df <- data.frame(feature    = names(mean_shap),
                       importance = as.numeric(mean_shap))
  bar_df <- tail(bar_df[order(bar_df$importance), ], 10)
  bar_df$feature <- factor(bar_df$feature, levels = feat_levels)

  g_bar <- ggplot(bar_df, aes(x = importance, y = feature, fill = importance)) +
    geom_col(width = 0.7) +
    geom_text(aes(label = sprintf("%.3f", importance)), hjust = -0.1, size = 3.8) +
    scale_fill_gradient(low = "#c6dbef", high = "#08519c") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.25))) +
    labs(x = "Mean |SHAP|", y = NULL) +
    theme_bw(base_size = 13) +
    theme(strip.background = element_rect(fill = "grey92"),
          axis.text.y      = element_blank(),
          axis.ticks.y     = element_blank(),
          axis.text.x      = element_text(size = 11),
          axis.title.x     = element_text(size = 12),
          legend.position  = "none")

  # ── combine with shared y-axis ────────────────────────────────────────────
  g_combined <- g_bee + g_bar + plot_layout(widths = c(2, 1))

  ggsave(file.path(OUT, combined_file), g_combined, width = 12, height = 5)
  cat(combined_file, "saved\n")
}

make_shap_plots("ela",      "10_shap_ela.pdf")
make_shap_plots("combined", "12_shap_combined.pdf")

cat("\nAll PDFs saved to:", OUT, "\n")
