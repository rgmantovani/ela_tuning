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
                              levels = c("Baseline", "Ela_metadataset"),
                              labels = c("Baseline", "ELA Meta-dataset"))

theme_pub <- theme_bw(base_size = 11) +
  theme(strip.background = element_rect(fill = "grey92"),
        legend.position  = "bottom",
        plot.title       = element_text(face = "bold", size = 11))

# ── 1. Violin + boxplot: AUC by dataset ──────────────────────────────────────
g1 <- ggplot(all_data, aes(x = dataset, y = AUC, fill = dataset)) +
  geom_violin(alpha = 0.4, trim = FALSE) +
  geom_boxplot(width = 0.15, outlier.size = 0.8) +
  scale_fill_manual(values = c("Baseline" = "#4e79a7", "ELA Meta-dataset" = "#f28e2b")) +
  labs(x = NULL, y = "AUC") +
  theme_pub + theme(legend.position = "none")

ggsave(file.path(OUT, "1_auc_by_dataset.pdf"), g1, width = 5, height = 4)
cat("1_auc_by_dataset.pdf saved\n")

# ── 2. Paired dot plot: ELA vs Baseline per algorithm × threshold ─────────────
agg <- aggregate(AUC ~ algorithm + threshold + dataset, data = all_data, FUN = mean)
agg_wide <- reshape(agg, idvar = c("algorithm", "threshold"),
                    timevar = "dataset", direction = "wide")
colnames(agg_wide)[colnames(agg_wide) == "AUC.Baseline"]         <- "baseline"
colnames(agg_wide)[colnames(agg_wide) == "AUC.ELA Meta-dataset"] <- "ela"

alg_shapes  <- c(DT = 15, KNN = 16, LR = 17, NB = 18, RF = 8, "SVM-Lin" = 10, "SVM-RBF" = 11, XGBoost = 12)
alg_colours <- RColorBrewer::brewer.pal(8, "Dark2")
names(alg_colours) <- levels(agg_wide$algorithm)

g2 <- ggplot(agg_wide, aes(x = baseline, y = ela, colour = algorithm, shape = algorithm)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey60", linewidth = 0.6) +
  geom_point(size = 2.8, alpha = 0.85) +
  scale_colour_manual(values = alg_colours, name = "Algorithm") +
  scale_shape_manual(values  = alg_shapes,  name = "Algorithm") +
  guides(colour = guide_legend(ncol = 1, override.aes = list(size = 3)),
         shape  = guide_legend(ncol = 1)) +
  labs(x = "Baseline AUC", y = "ELA Meta-dataset AUC") +
  coord_equal(xlim = c(0.50, 0.82), ylim = c(0.50, 0.82)) +
  theme_pub +
  theme(legend.position       = "right",
        legend.key.size       = unit(0.45, "cm"),
        legend.text           = element_text(size = 8),
        legend.title          = element_text(size = 8, face = "bold"),
        legend.box.background = element_blank())

ggsave(file.path(OUT, "2_paired_auc.pdf"), g2, width = 5.5, height = 4.5)
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
                       midpoint = 0.75, limits = c(0.50, 0.82)) +
  facet_wrap(~dataset) +
  labs(x = expression(tau), y = NULL, fill = "AUC") +
  theme_pub + theme(legend.position = "right")

ggsave(file.path(OUT, "4_heatmap_alg_threshold.pdf"), g4, width = 9, height = 4)
cat("4_heatmap_alg_threshold.pdf saved\n")

# ── 10–11. SHAP plots (requires 04_shap_analysis.py to be run first) ──────────
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

  g10 <- ggplot(shap_long, aes(x = shap_val, y = feature, colour = feat_val)) +
    geom_jitter(height = 0.22, size = 1.6, alpha = 0.7) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
    scale_colour_gradient2(low = "#d53e4f", mid = "#ffffbf", high = "#3288bd",
                           midpoint = 0, name = "Feature\nvalue\n(scaled)") +
    labs(x = "SHAP value (impact on model output)", y = NULL) +
    theme_pub +
    theme(axis.text.y    = element_text(family = "mono", size = 8.5),
          legend.position = "right")

  ggsave(file.path(OUT, "10_shap_beeswarm.pdf"), g10, width = 8.5, height = 5)
  cat("10_shap_beeswarm.pdf saved\n")

  bar_df <- data.frame(feature = names(mean_shap), importance = as.numeric(mean_shap))
  bar_df <- tail(bar_df[order(bar_df$importance), ], 10)
  bar_df$feature <- factor(bar_df$feature, levels = bar_df$feature)

  g11 <- ggplot(bar_df, aes(x = importance, y = feature, fill = importance)) +
    geom_col(width = 0.7) +
    geom_text(aes(label = sprintf("%.3f", importance)), hjust = -0.1, size = 3.2) +
    scale_fill_gradient(low = "#c6dbef", high = "#08519c") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
    labs(x = "Mean |SHAP value|", y = NULL) +
    theme_pub +
    theme(axis.text.y     = element_text(family = "mono", size = 8.5),
          legend.position = "none")

  ggsave(file.path(OUT, "11_shap_bar.pdf"), g11, width = 7.5, height = 4.5)
  cat("11_shap_bar.pdf saved\n")
} else {
  cat("SHAP files not found in tmp/ — run 04_shap_analysis.py first.\n")
}

cat("\nAll PDFs saved to:", OUT, "\n")
