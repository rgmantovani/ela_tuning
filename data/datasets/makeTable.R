#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

# files = list.files()
# target = lapply(files, function(file) {
#   data = RWeka::read.arff(file)
#   tar = colnames(data)[ncol(data)]
#   print(file)
#   return(tar)
#   print("==========")
# }) 


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

files = list.files(path = "preprocessed")
subs = strsplit(files, "_")

ids = as.numeric(unique(unlist(lapply(subs, function(elem) return(elem[1])))))

datasets = read.csv(file = "old_datasets.csv")
all.ids = unique(c(ids, datasets$OpenML.id))

df.data = OpenML::listOMLDataSets()

sel.data = df.data[which(df.data$data.id %in% all.ids), ]

tab = sel.data[,c(1,2,12,11,10,7,9)]

# write.csv(tab, file = "table.csv")

# setdiff(all.ids, sel$data.id)
# 173 40474 40475 40499 40496

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
