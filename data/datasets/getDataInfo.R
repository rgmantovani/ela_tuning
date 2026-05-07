datasets = list.files("preprocessed/")

avail.data = OpenML::listOMLDataSets(limit = 10000)

ids = strsplit(x = datasets, split = "_")
ids = unlist(lapply(ids, function(elem) return(elem[1])))
ids = as.numeric(ids)

common = intersect(ids, avail.data$data.id)

cols = c("name", "data.id", "version", "number.of.features", "number.of.instances", "number.of.classes", "majority.class.size", 
  "minority.class.size")
sub = avail.data[which(avail.data$data.id %in% ids), cols]

sub$minmaj = sub$minority.class.size/sub$majority.class.size
write.csv(sub, file = "sub.csv")
# > setdiff(ids, avail.data$data.id)
# 173 40474 40475 40496 40499 40733 40734 40735 40736

# minority.class.size/majority.class.size