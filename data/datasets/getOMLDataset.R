  
library("OpenML")

ids = c(3, 6, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 28, 29, 30, 31, 32, 35, 36, 37, 38, 40, 43, 44, 46, 48, 49, 50, 51, 53, 54, 
  55, 56, 59, 60, 61, 164, 182, 185, 187, 188, 292, 294, 299, 300, 301, 307, 310, 311, 312, 316, 329, 333, 334, 335, 336, 338, 375, 377, 
  378, 444, 446, 448, 450, 451, 452, 455, 458, 461, 463, 464, 466, 469, 470, 475, 481, 679, 685, 694, 1037, 1038, 1040, 1041, 1042, 1043, 
  1044, 1046, 1048, 1049, 1050, 1053, 1054, 1056, 1061, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1071, 1073, 1075, 1100, 1115, 1116, 1120, 
  1121, 1167, 1442, 1443, 1444, 1446, 1447, 1451, 1452, 1453, 1455, 1456, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1470, 
  1471, 1473, 1475, 1476, 1477, 1479, 1480, 1484, 1485, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 40877, 40878, 
  1501, 1504, 1506, 1507, 1508, 1510, 1511, 1512, 1514, 1519, 1520, 1523, 1524, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 
  1536, 1538, 1539, 1540, 1541, 1547, 1548, 1549, 1551, 1552, 1553, 1554, 1555, 1556, 1558, 1559, 1565, 1566, 1568, 1570, 1600, 4534, 4538, 
  4550, 6332, 40474, 40475, 40496, 40499, 40733)

# ids = c(42, 151, 478, 1036, 1112, 1114, 1176, 1478, 1486, 1505, 1515, 1590, 4134, 4135, 23380, 23381, 23512, 40509, 40536)

dir.create(path = "raw", showWarnings=FALSE)

aux = lapply(ids, function(id) {
  cat("==============================\n")
  dataset = getOMLDataSet(data.id = id)
  cat(" dataset: ", id, " - ", dataset$desc$name, "\n")
  new.name = paste0("raw/", dataset$desc$id, "_", dataset$desc$name, ".arff")

  if(file.exists(new.name)) {
    cat("skipping ... \n")
  } else {
    temp = dataset$data
    if(!is.na(dataset$desc$row.id.attribute)) {
      cat(" - row id to remove\n")
      print(dataset$desc$row.id.attribute)
      rm.id = which(colnames(temp) == dataset$desc$row.id.attribute)
      temp = temp[,-rm.id]
    }


    cat(" - reordering attrs\n")
    target.id = which(colnames(temp) == dataset$target.features)

    if(length(target.id)== 0) {
      target.id = which(colnames(temp) == "Class")
    }

    temp = cbind(temp[,-target.id], temp[,target.id])
    colnames(temp)[ncol(temp)] = "Class"
    temp$Class = as.factor(as.character(temp$Class))

    cat(" - saving file\n")
    RWeka::write.arff(x = temp, file = new.name)
  }

})
