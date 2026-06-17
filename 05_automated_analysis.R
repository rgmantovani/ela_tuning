library(ggplot2)
library(reshape2)

OUT <- "./plots/analysis"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ── Load and aggregate results ────────────────────────────────────────────────
files <- list.files("results", pattern = "loo_metrics_seed_", full.names = TRUE)
all_data <- do.call(rbind, lapply(files, read.csv))

# Standardise column names
colnames(all_data) <- c("dataset", "algorithm", "threshold", "NumFeatures", "F1", "BAC", "AUC")

all_data$threshold <- factor(all_data$threshold)

alg_order <- c("DecisionTree", "KNN", "LogisticRegression", "NaiveBayes",
                "RandomForest", "SVM_Linear", "SVM_RBF", "XGBoost")
alg_labels <- c("DT", "KNN", "LR", "NB", "RF", "SVM-Lin", "SVM-RBF", "XGBoost")

all_data$algorithm  <- factor(all_data$algorithm,  levels = alg_order, labels = alg_labels)
all_data$dataset    <- factor(all_data$dataset,
                              levels = c("Baseline", "Ela_metadataset"),
                              labels = c("Baseline", "ELA Metadataset"))

theme_pub <- theme_bw(base_size = 11) +
  theme(strip.background = element_rect(fill = "grey92"),
        legend.position = "bottom",
        plot.title = element_text(face = "bold", size = 11))

# ── 1. Violin + boxplot: AUC by dataset ──────────────────────────────────────
g1 <- ggplot(all_data, aes(x = dataset, y = AUC, fill = dataset)) +
  geom_violin(alpha = 0.4, trim = FALSE) +
  geom_boxplot(width = 0.15, outlier.size = 0.8) +
  scale_fill_manual(values = c("Baseline" = "#4e79a7", "ELA Metadataset" = "#f28e2b")) +
  labs(title = "AUC Distribution by Meta-dataset", x = NULL, y = "AUC") +
  theme_pub + theme(legend.position = "none")

ggsave(file.path(OUT, "1_auc_by_dataset.pdf"), g1, width = 5, height = 4)
cat("1_auc_by_dataset.pdf saved\n")

# ── 2. Paired dot plot: ELA vs Baseline per algorithm × threshold ─────────────
agg <- aggregate(AUC ~ algorithm + threshold + dataset, data = all_data, FUN = mean)
agg_wide <- reshape(agg, idvar = c("algorithm", "threshold"),
                    timevar = "dataset", direction = "wide")
colnames(agg_wide)[colnames(agg_wide) == "AUC.Baseline"]        <- "baseline"
colnames(agg_wide)[colnames(agg_wide) == "AUC.ELA Metadataset"] <- "ela"

g2 <- ggplot(agg_wide, aes(x = baseline, y = ela, colour = algorithm, shape = threshold)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey50") +
  geom_point(size = 2.5, alpha = 0.9) +
  scale_colour_brewer(palette = "Dark2") +
  labs(title = "Paired AUC: ELA vs Baseline",
       x = "Baseline AUC", y = "ELA Metadataset AUC",
       colour = "Algorithm", shape = expression(tau)) +
  coord_equal(xlim = c(0.50, 0.82), ylim = c(0.50, 0.82)) +
  theme_pub

ggsave(file.path(OUT, "2_paired_auc.pdf"), g2, width = 6, height = 5)
cat("2_paired_auc.pdf saved\n")

# ── 3. Grouped barplot with error bars: AUC by algorithm ─────────────────────
agg2 <- aggregate(AUC ~ algorithm + dataset, data = all_data,
                  FUN = function(x) c(mean = mean(x), sd = sd(x)))
agg2 <- do.call(data.frame, agg2)
colnames(agg2)[3:4] <- c("mean", "sd")

g3 <- ggplot(agg2, aes(x = algorithm, y = mean, fill = dataset, ymin = mean - sd, ymax = mean + sd)) +
  geom_col(position = position_dodge(0.75), width = 0.7) +
  geom_errorbar(position = position_dodge(0.75), width = 0.25) +
  scale_fill_manual(values = c("Baseline" = "#4e79a7", "ELA Metadataset" = "#f28e2b")) +
  labs(title = "Mean AUC ± SD by Algorithm", x = NULL, y = "AUC", fill = NULL) +
  coord_cartesian(ylim = c(0.50, 0.84)) +
  theme_pub + theme(axis.text.x = element_text(angle = 35, hjust = 1))

ggsave(file.path(OUT, "3_auc_by_algorithm.pdf"), g3, width = 7, height = 4.5)
cat("3_auc_by_algorithm.pdf saved\n")

# ── 4. Heatmap: algorithm × threshold, faceted by dataset ────────────────────
agg3 <- aggregate(AUC ~ algorithm + threshold + dataset, data = all_data, FUN = mean)

g4 <- ggplot(agg3, aes(x = threshold, y = algorithm, fill = AUC)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%.3f", AUC)), size = 2.8) +
  scale_fill_gradient2(low = "#d73027", mid = "#ffffbf", high = "#1a9850",
                       midpoint = 0.70, limits = c(0.50, 0.82)) +
  facet_wrap(~dataset) +
  labs(title = "Mean AUC: Algorithm × Threshold", x = expression(tau), y = NULL, fill = "AUC") +
  theme_pub + theme(legend.position = "right")

ggsave(file.path(OUT, "4_heatmap_alg_threshold.pdf"), g4, width = 9, height = 4)
cat("4_heatmap_alg_threshold.pdf saved\n")

# ── 5. Line + ribbon: AUC vs threshold per algorithm ─────────────────────────
agg4 <- aggregate(AUC ~ algorithm + threshold + dataset, data = all_data,
                  FUN = function(x) c(mean = mean(x), sd = sd(x)))
agg4 <- do.call(data.frame, agg4)
colnames(agg4)[4:5] <- c("mean", "sd")
agg4$threshold <- as.numeric(as.character(agg4$threshold))

g5 <- ggplot(agg4, aes(x = threshold, y = mean, colour = algorithm,
                        fill = algorithm, ymin = mean - sd, ymax = mean + sd)) +
  geom_ribbon(alpha = 0.12, colour = NA) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  scale_colour_brewer(palette = "Dark2") +
  scale_fill_brewer(palette = "Dark2") +
  scale_x_continuous(breaks = c(0.80, 0.85, 0.90, 0.95)) +
  facet_wrap(~dataset) +
  labs(title = "AUC vs Correlation Threshold",
       x = expression(paste("Correlation threshold (", tau, ")")),
       y = "Mean AUC", colour = "Algorithm", fill = "Algorithm") +
  theme_pub

ggsave(file.path(OUT, "5_auc_vs_threshold.pdf"), g5, width = 9, height = 4.5)
cat("5_auc_vs_threshold.pdf saved\n")

# ── 6. Facet grid: all 3 metrics by algorithm ─────────────────────────────────
long <- melt(all_data, measure.vars = c("F1", "BAC", "AUC"),
             variable.name = "metric", value.name = "value")

g6 <- ggplot(long, aes(x = algorithm, y = value, fill = dataset)) +
  geom_boxplot(outlier.size = 0.5, linewidth = 0.4) +
  scale_fill_manual(values = c("Baseline" = "#4e79a7", "ELA Metadataset" = "#f28e2b")) +
  facet_wrap(~metric, ncol = 1, scales = "free_y") +
  labs(title = "All Metrics by Algorithm and Dataset", x = NULL, y = NULL, fill = NULL) +
  theme_pub + theme(axis.text.x = element_text(angle = 35, hjust = 1),
                    legend.position = "bottom")

ggsave(file.path(OUT, "6_all_metrics_facet.pdf"), g6, width = 8, height = 9)
cat("6_all_metrics_facet.pdf saved\n")

# ── 7. Seed stability: AUC per seed, stochastic algorithms only ───────────────
stochastic <- c("DT", "RF", "SVM-Lin", "SVM-RBF")
seed_data <- all_data[all_data$algorithm %in% stochastic, ]

g7 <- ggplot(seed_data, aes(x = algorithm, y = AUC, fill = dataset)) +
  geom_boxplot(outlier.size = 0.6, linewidth = 0.4) +
  scale_fill_manual(values = c("Baseline" = "#4e79a7", "ELA Metadataset" = "#f28e2b")) +
  labs(title = "Seed Stability: AUC Variation per Algorithm",
       x = NULL, y = "AUC", fill = NULL) +
  theme_pub

ggsave(file.path(OUT, "7_seed_stability.pdf"), g7, width = 6.5, height = 4)
cat("7_seed_stability.pdf saved\n")

cat("\nAll PDFs saved to:", OUT, "\n")

# ── 8. Top-10 XGBoost feature importances (best setup: ELA, tau=0.95) ─────────
feat_imp <- data.frame(
  feature = c("nbc.nn_nb.dist.ratio.sd", "fdc.slope", "ela_ic.h_max",
               "gh.norm.sd", "gh.norm.mean", "ela_lc.y_skewness",
               "ela_meta.quad_simple.adj_r2", "ls.mean",
               "ela_lc.r_pearson_dist_centroid", "ela_meta.lin_simple.adj_r2"),
  importance = c(0.125433, 0.059522, 0.056116, 0.051004, 0.049919,
                 0.044120, 0.039358, 0.039014, 0.036488, 0.031702)
)
feat_imp$feature <- factor(feat_imp$feature,
                            levels = feat_imp$feature[order(feat_imp$importance)])

g8 <- ggplot(feat_imp, aes(x = importance, y = feature, fill = importance)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.4f", importance)), hjust = -0.1, size = 3.2) +
  scale_fill_gradient(low = "#c6dbef", high = "#08519c") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title = "Top-10 ELA Features — XGBoost (ELA Metadataset, τ = 0.95)",
       x = "Feature Importance (gain)", y = NULL) +
  theme_pub + theme(legend.position = "none",
                    axis.text.y = element_text(family = "mono", size = 9))

ggsave(file.path(OUT, "8_xgboost_top10_features.pdf"), g8, width = 7.5, height = 4.5)
cat("8_xgboost_top10_features.pdf saved\n")

# ── 9–10. SHAP plots (requires 05_shap_analysis.py to be run first) ───────────
SHAP_VALUES <- "tmp/shap_values.csv"
SHAP_X      <- "tmp/shap_X.csv"

if (file.exists(SHAP_VALUES) && file.exists(SHAP_X)) {
  shap <- read.csv(SHAP_VALUES)
  X    <- read.csv(SHAP_X)

  mean_shap   <- sort(colMeans(abs(shap)), decreasing = FALSE)
  top10_names <- tail(names(mean_shap), 10)

  shap_long <- data.frame()
  for (f in top10_names) {
    shap_long <- rbind(shap_long, data.frame(
      feature  = f,
      shap_val = shap[[f]],
      feat_val = scale(X[[f]])[, 1]
    ))
  }
  shap_long$feature <- factor(shap_long$feature, levels = top10_names)

  g9 <- ggplot(shap_long, aes(x = shap_val, y = feature, colour = feat_val)) +
    geom_jitter(height = 0.22, size = 1.6, alpha = 0.7) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
    scale_colour_gradient2(low = "#3288bd", mid = "#ffffbf", high = "#d53e4f",
                           midpoint = 0, name = "Feature\nvalue\n(scaled)") +
    labs(title = "SHAP beeswarm — XGBoost (ELA Metadataset, τ = 0.95)",
         x = "SHAP value (impact on model output)", y = NULL) +
    theme_pub +
    theme(axis.text.y = element_text(family = "mono", size = 8.5),
          legend.position = "right")

  ggsave(file.path(OUT, "10_shap_beeswarm.pdf"), g9, width = 8.5, height = 5)
  cat("10_shap_beeswarm.pdf saved\n")

  bar_df <- data.frame(feature = names(mean_shap), importance = as.numeric(mean_shap))
  bar_df <- tail(bar_df[order(bar_df$importance), ], 10)
  bar_df$feature <- factor(bar_df$feature, levels = bar_df$feature)

  g10 <- ggplot(bar_df, aes(x = importance, y = feature, fill = importance)) +
    geom_col(width = 0.7) +
    geom_text(aes(label = sprintf("%.3f", importance)), hjust = -0.1, size = 3.2) +
    scale_fill_gradient(low = "#c6dbef", high = "#08519c") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
    labs(title = "Mean |SHAP| — top 10 features",
         x = "Mean |SHAP value|", y = NULL) +
    theme_pub +
    theme(axis.text.y = element_text(family = "mono", size = 8.5),
          legend.position = "none")

  ggsave(file.path(OUT, "11_shap_bar.pdf"), g10, width = 7.5, height = 4.5)
  cat("11_shap_bar.pdf saved\n")
} else {
  cat("SHAP files not found in tmp/ — run 05_shap_analysis.py first.\n")
}
