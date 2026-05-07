#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

# Pre-processing function
# - remove constant attributes
# - rename categorical attributes
# - normalize all attribues (mean=0, std=1), but Class
# - remove duplicate examples
# - format file header

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

library("mlr")
library("RSNNS")

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------


preprocessing = function(file) {

  cat("* File:",file,"\n")

  if(file.exists(file = paste0("preprocessed/",file))) {
    cat(" - skipping: already done\n")
  } else {

    # read the file
    old.data = foreign::read.arff(file = paste0("original/", file))
    data = old.data

    # Doing imputation
    if(any(is.na(data))) {
      cat(" - doing imputation\n")
      imp = mlr::impute(obj = data, target = "Class", 
        classes = list(integer = mlr::imputeMedian(), 
        factor  = mlr::imputeConstant(const = "New"), 
        numeric = mlr::imputeMedian())
      )
      data = imp$data
    }
    
    data = mlr::createDummyFeatures(obj = data, target = "Class", method = "1-of-n")
    target.id = which(colnames(data) == "Class")

    class = data[,target.id]
    data  = data[,-target.id]

    # remove constant attributes
    for(i in colnames(data)) {  

      # categorical attributes
      if(is.factor(data[,i])) {
        
        if(nlevels(data[,i]) == 1 || nlevels(data[,i]) == nrow(data)) {
          data[,i] = NULL
        }else{
          #renaming predicates
          data[,i] = factor(data[,i])
          levels(data[,i]) = factor(1:length(levels(data[,i])))
        }
      # numeric attributes  
      } else {
        if(sd(data[,i]) == 0) {
          data[,i] = NULL
        } else {
          data[,i] = RSNNS::normalizeData(data[,i], type="norm")
        }
      }
    }

    data = cbind(data, class)

    # remove duplicated examples
    aux = which(duplicated(data))
    if(length(aux) != 0){
      data = data[-aux,]
    }

    # format the header
    # colnames(data) = c(paste("V", rep(1:(ncol(data)-1)), sep=""), "Class")
    colnames(data)[ncol(data)] = "Class"
    data$Class = factor(data$Class)
    rownames(data) = NULL

    # save the arff file
    foreign::write.arff(data, paste0("preprocessed/",file))
  }

}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

preprocessData = function(){
  files = list.files(path = "original/")
  for(i in 1:length(files)) {
    cat("----------\n")
    cat(i,"/", length(files), "")
    file = files[i]
    preprocessing(file)
  }
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

preprocessData()

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------