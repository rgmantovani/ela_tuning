# ---------------------------------
# ---------------------------------

library(reshape2)
library(ggplot2)

# ---------------------------------
# Plotting Hyperspaces
# ---------------------------------

dir.create("./plots/hyerspaces", showWarnings = FALSE)
LANDSCAPE_PATH = "data/landscapeInputs/classif.svm/"

land_files = list.files(path = LANDSCAPE_PATH)

for(file in land_files) {

  print(file)
  data = read.csv(paste0(LANDSCAPE_PATH, file))
  colnames(data)[3] = "BER"
  data$BAC = 1 - data$BER

  g = ggplot(data, mapping = aes(x = cost, y = gamma, colour = BAC)) 
  g = g + geom_point(size = 0.1)
  g = g + scale_colour_gradient2( low = "red", high = "blue", mid = "white", 
    midpoint = 0.5)
  g = g + labs(x = "log2(cost)", y = "log2(gamma)")
  # g

  outputile = paste0("./plots/hyerspaces/", gsub(x = file, pattern = ".csv", replacement = "_hyperspace.pdf" ))
  ggsave(g, file = outputile, width = 3.49, height = 2.56)

}


# ---------------------------------
# ---------------------------------