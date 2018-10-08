library(tm)
library(SnowballC)
library(wordcloud)
library(tidyverse)
library(tidytext)
library(topicmodels)


post.data <- read.csv('..\\dataframes\\post_data.csv')
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

test.entries = as.vector(unlist(post.clean.test.data, use.names=FALSE))

test.entries = test.entries[-which(as.numeric(as.vector(test.entries)) == 0)]



test.corpus <- Corpus(VectorSource(test.entries))

#test.corpus <- tm_map(test.corpus, removeNumbers)
test.corpus <- tm_map(test.corpus, content_transformer(tolower))
test.corpus <- tm_map(test.corpus, removePunctuation)
test.corpus <- tm_map(test.corpus, stripWhitespace)
test.corpus <- tm_map(test.corpus, removeWords, c('the', 'and', stopwords('english')))
test.corpus <- tm_map(test.corpus, stemDocument, language = 'english')



writeLines(as.character(test.corpus[[21]]))
dtm = DocumentTermMatrix(test.corpus, control = list(weighting = weightTf))
freq = colSums(as.matrix(dtm))
length(freq)
ord = order(freq, decreasing = TRUE)
freq[ord]

write.csv(freq[ord], "word_freq.csv")

#Code template for LDA: https://eight2late.wordpress.com/2015/09/29/a-gentle-introduction-to-topic-modeling-using-r/

#Set parameters for Gibbs sampling
burnin <- 4000
iter <- 2000
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE

#Number of topics
k <- 3

#Run LDA using Gibbs sampling
ldaOut <-LDA(dtm,k, method='Gibbs' , control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))

ldaOut.topics = as.matrix(topics(ldaOut))
ldaOut.terms = as.matrix(terms(ldaOut, 10))

topicProbabilities = as.data.frame(ldaOut@gamma)
topicProbabilities$Topic = ldaOut.topics
topicProbabilities$Docs = test.entries
write.csv(topicProbabilities, "3_topic_LDA_tfIDF.csv")




# cluster documents in topic space
document.topic.probabilities = ldaOut@gamma  # topic distribution for each document
topic.space.kmeans.clusters = kmeans(document.topic.probabilities, 3)
topic.space.clustered.news = split(test.corpus, topic.space.kmeans.clusters$cluster)
topic.space.clustered.news[[1]][[99]]$content

document.vector = c()
topic.vector = c()
for (i in 1:length(topic.space.clustered.news)){
  for(j in 1:length(topic.space.clustered.news[[i]])){
    document.vector = c(document.vector, topic.space.clustered.news[[i]][[j]]$content)
    topic.vector = c(topic.vector, i)
  }
  }

df.kmeans = data.frame("topic" = topic.vector, "document" = document.vector)

write.csv(df.kmeans, 'k_means_clustering.csv')

