#setwd('C:\\Users\\vaibhav\\Documents\\UVA\\Fall\\Capstone\\Code\\ARL\\data_processing')
library(tm)
library(SnowballC)
library(wordcloud)

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

#Creating a corpus from the testing data
test.corpus <- Corpus(VectorSource(test.entries))

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
test.corpus.dtm

#Inspecting the first 5 documents and the first 5 words in the corpus
inspect(test.corpus.dtm[1:5, 1:5])

#Getting the 5 most frequent words
freq <- data.frame(sort(colSums(as.matrix(test.corpus.dtm)), decreasing=TRUE))
freq.df <- head(freq, 5)
colnames(freq.df) <- c('frequency')

#Barplot to get most frequent words
barplot(freq.df$frequency, names.arg=rownames(freq.df), col=c("beige","orange"), ylab="Count of words", ylim = c(0, 160), main = 'Most frequent words')
