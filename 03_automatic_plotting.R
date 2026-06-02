library(reshape2)
library(ggplot2)

# ---------------------------------
# Plotting Hyperspaces
# ---------------------------------

dir.create("./plots/hyperspaces", showWarnings = FALSE)
LANDSCAPE_PATH <- "data/landscapeInputs/classif.svm/"

land_files <- list.files(path = LANDSCAPE_PATH)

for (file in land_files) {

  print(file)
  data <- read.csv(paste0(LANDSCAPE_PATH, file))
  colnames(data)[3] <- "BER"
  data$BAC <- 1 - data$BER

  g <- ggplot(data, mapping = aes(x = cost, y = gamma, colour = BAC))
  g <- g + geom_point(size = 0.1)
  g <- g + scale_colour_gradient2(low = "red", high = "blue", mid = "white", midpoint = 0.5)
  g <- g + labs(x = "log2(cost)", y = "log2(gamma)")

  output_file <- paste0("./plots/hyperspaces/",
    gsub(x = file, pattern = ".csv", replacement = "_hyperspace.pdf", fixed = TRUE))
  ggsave(g, file = output_file, width = 3.49, height = 2.56)

}

# ---------------------------------
# Aggregating results
# ---------------------------------

results_files <- list.files(path = "results", pattern = "loo_metrics_", full.names = TRUE)

aux_res <- lapply(results_files, read.csv)

aux_perf <- lapply(aux_res, function(res) res[4:7])

avg_perf <- Reduce("+", aux_perf) / length(aux_perf)
colnames(avg_perf) <- paste0("average_", colnames(avg_perf))

sd_perf <- data.frame(apply(sapply(aux_perf, as.matrix, simplify = "array"), c(1, 2), sd))
colnames(sd_perf) <- paste0("sd_", colnames(sd_perf))

df_full <- cbind(aux_res[[1]][1:4], avg_perf, sd_perf)
df_full <- df_full[order(df_full$average_AUC, decreasing = TRUE), ]
