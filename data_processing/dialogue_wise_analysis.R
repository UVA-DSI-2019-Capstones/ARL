# setwd('C:\\Users\\vaibhav\\Documents\\UVA\\Fall\\Capstone\\Code\\ARL\\data_processing')
library(tm)
library(SnowballC)
library(wordcloud)
library(tidyverse)
library(tidytext)


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

#Getting overall score dataframe of each author
overall.score <- subset(post.data[19,], select = names(post.data) %ni% nums)



#Extracting all the entries from the data frame 
test.entries <- as.vector(unlist(post.clean.test.data))
rating.entries <- as.vector(unlist(post.clean.rating.data))

#Extracting all the numerical entries from the test data
numerical.indices <- c()
for (i in 1:length(test.entries)) {
  if (vector.is.numeric(test.entries[i]) == TRUE) {
    numerical.indices <- c(numerical.indices, i)
  }
}

#Removing the numerical entries from the test data and their corresponding ratings
test.entries <- test.entries[-numerical.indices]
rating.entries <- rating.entries[-numerical.indices]

master.df <- data.frame(matrix(ncol = 4, nrow = 0))
x <- c("document", "term", "count", "rating")
colnames(master.df) <- x

for (i in 1 : length(test.entries)) {
  
  test.corpus <- Corpus(VectorSource(test.entries[i]))
  
  "
  Next we normalize the texts in the reviews using a series of pre-processing steps: 
  1. Switch to lower case 
  2. Remove punctuation marks 
  3. Remove extra whitespaces
  4. Remove stop words
  5. Stemmatize the words
  "
  test.corpus <- tm_map(test.corpus, content_transformer(tolower))
  test.corpus <- tm_map(test.corpus, removePunctuation)
  test.corpus <- tm_map(test.corpus, stripWhitespace)
  test.corpus <- tm_map(test.corpus, removeWords, c("the", "and", stopwords("english")))
  test.corpus <- tm_map(test.corpus, stemDocument, language = "english")
  
  "
  To analyze the textual data, we use a Document-Term Matrix (DTM) representation: documents as the rows, terms/words as the columns, frequency of the term in the document as the entries. Because the number of unique words in the corpus the dimension can be large.
  "
  test.corpus.dtm <- DocumentTermMatrix(test.corpus)
  
  new.entry <- cbind(data.frame(tidy(test.corpus.dtm)), data.frame('rating' = rating.entries[i]))
  master.df <- rbind(master.df, new.entry)
}