library(BMA)

data = read.csv('D:/Study/Chuyên đề NC/source_code/ResearchTopic/Malware_dataset.csv')

Y <- factor(data$classification)
X <- data[, !(names(data) %in% "hash")]

X <- lapply(X, factor)

#levels_list <- sapply(X, levels)

#print(levels_list)

#for (col_name in names(X)) {
#  cat("Cột:", col_name, "\nLevels:", toString(levels(X[[col_name]])), "\n\n")
#}

X <- as.data.frame(X)
X_filtered <- X[, sapply(X, function(x) length(levels(x)) > 1)]
print(dim(X_filtered))
search = bicreg(X_filtered, Y, strict = FALSE, OR = 20)
summary(search)