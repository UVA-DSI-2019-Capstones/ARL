#setwd('C:\\Users\\vaibhav\\Documents\\UVA\\Fall\\Capstone\\Code\\ARL\\data_processing')

post.data <- read.csv('dataframes\\post_data.csv')
post.rating.data <- data.frame()
post.test.data <- data.frame()

vector.is.numeric <- function(x) {
  #Function that gives true if there are only numeric values
  return(!anyNA(as.numeric(as.character(x))))
}

for (i in 1: 18) {
  if (i %% 2 == 1) {
    post.rating.data <- rbind(post.rating.data, as.data.frame(post.data[i,]))
  } else {
    post.test.data <- rbind(post.test.data, as.data.frame(post.data[i,]))
  }
}

#Finding which columns are numeric
nums <- unlist(lapply(post.test.data, vector.is.numeric)) 
nums <- names(which(nums == TRUE))

#Removing all the numeric columns from the test data
`%ni%` <- Negate(`%in%`)
post.clean.test.data <- subset(post.test.data, select = names(post.test.data) %ni% nums)

#Removing all the columns from the rating data that were removed from the test data
post.clean.rating.data <- subset(post.rating.data, select = names(post.rating.data) %ni% nums)
