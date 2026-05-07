
INPUT_CSV = "data/classif_svm_169d_95_average.arff"
LABELS_CSV = "results/df_gain_complete.csv"


df_features = farff::readARFF("data/classif_svm_169d_95_average.arff") 
df_gain     = read.csv( "results/df_gain_complete.csv")

df_gain$X = NULL

aux = lapply(seq_len(nrow(df_gain)), function(i) {
    elem = df_gain[i, ]
    # label =  names(which.max(elem[2:5])[1]) # including optimized
    label =  names(which.max(elem[3:5])[1])  # not using optimized, just defaults
    return(label)
})

colnames(df_features)[1] = "dataset"

df_gain$target = unlist(aux)

# df_features$Class = Class
metadataset = merge(df_features, df_gain[,c("dataset", "target")], by = "dataset")
metadataset$Class = NULL
write.csv(metadataset, file = "data/metadataset.csv", row.names = FALSE)