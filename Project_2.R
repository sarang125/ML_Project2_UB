"This is the code for extracting the UB related tweets from tweeter 
and performing a class prediction task based on its emergency(1) or no-emergency(0) nature
"
# Step 1 : Installing the necessary packages for extracting..
#...and cleaning the tweet data using twitteR and tm package in R

#install.packages('twitteR')
library(twitteR)
#install.packages('rtweet')
library(rtweet)
#install.packages('tm')
library(tm)

#install.packages("devtools")
#devtools::install_github("mkearney/rtweet")

# Step 2 : Establishing the one-time connection with Twitter

# The keys have been hashed from security point of view

api_key <- '***'

api_secret <- '***'

access_token <- '***'

access_token_secret <- '***'

# Authentication for extracting the tweets
setup_twitter_oauth(api_key, api_secret, access_token = access_token, access_secret = access_token_secret)

# Step 3 : Extracting the relavant tweets

# First set of train data from UBuffalo handle mostly the label '0' data
ub_tweets_1 <- userTimeline('UBuffalo', n = 500)
length(ub_tweets) # We could fetch 42 only

ub_tweets_2 <- userTimeline('UBStudentExp', n = 500)
length(ub_tweets_2) # We could fetch 89 only

ub_tweets_3 <- userTimeline('UBAthletics', n = 500)
length(ub_tweets_3)# We could fetch 30 only

ub_tweets_4 <- userTimeline('ubalumni', n = 500)
length(ub_tweets_4) # We could fetch 251 only

ub_tweets_5 <- userTimeline('UBCommunity', n = 500)
length(ub_tweets_5) # We could fetch 92 only

# This mostly has the emergency data with Police force as first respondents
ub_tweets_6 <- userTimeline('UBuffaloPolice', n = 3000)
length(ub_tweets_6) # We could fetch 452 only

# By defalut the class is list, so converting into Dataframe
df1 <- twListToDF(ub_tweets_1)
df2 <- twListToDF(ub_tweets_2)
df3 <- twListToDF(ub_tweets_3)
df4 <- twListToDF(ub_tweets_4)
df5 <- twListToDF(ub_tweets_5)
df6 <- twListToDF(ub_tweets_6)
write.csv(df6,file = 'pos_tweets.csv')
df6 <- read.csv('filtered_positive_incidents.csv')

head(df6$text)

# Combining the train and test datasets
# Since we have lesser no. of positive labelled instances we restrict
#..the negative ones too

df <- rbind(df1,df2,df3,df6)
dim(df)

# Appending the labels
df$label <- c(rep(0,(nrow(df)-nrow(df6))),rep(1,nrow(df6)))
dim(df)
head(df)
colnames(df)

write.csv(df, 'DatasetUsed.csv')

final_df <- df[c('text','label')]
dim(final_df)
table(final_df$label) # 161 Negative cases (Non-emergency) and 15 Positive cases(Emergency cases)

# Step 4 : Perfrming the Text Mining on the "text" column of the tweets with tm library

# Building the corpus
df_corpus <- Corpus(VectorSource(final_df$text))

# Cleaning up the tweets

# Removing url using function
removeURL <- function(x) gsub('http[^[:space:]]*','',x)
df_corpus <- tm_map(df_corpus, content_transformer(removeURL))

# Retaining only the alphabets and space
removeExtra <- function(x) gsub('[^[:alpha:][:space:]]*','',x)
df_corpus <- tm_map(df_corpus, content_transformer(removeExtra))
df_corpus <- tm_map(df_corpus, tolower)
df_corpus <- tm_map(df_corpus, removePunctuation)
df_corpus <- tm_map(df_corpus, removeNumbers)
df_corpus <- tm_map(df_corpus, removeWords, stopwords(kind = 'en'))

# Visualize the content and creating the Document Term matrix for easy data handling
content(df_corpus)

df_dtm <- DocumentTermMatrix(df_corpus)
df_dtm_m <- as.matrix(df_dtm)
head(df_dtm_m,1)
class(df_dtm_m)
dim(df_dtm_m)

final_dataset <- cbind(df_dtm_m,c(rep(0,(nrow(df)-nrow(df6))),rep(1,nrow(df6))))
dim(final_dataset)


# Step 5 : Performing Gradient Descent optimization using the built-in package
library(gradDescent)

# Let's track time to run
devtools::install_github("collectivemedia/tictoc")
library(tictoc)

Splited_set <- splitData(final_dataset, dataTrainRate = 0.8, seed = 123)

dim(Splited_set$dataTrain) # (140 Documents(instances) by 816 (Terms))
dim(Splited_set$dataTest) # (36 by 816)
dim(t(Splited_set$dataTest)) # Performing row-column transformation

tic('Start run')
grad_descent <- GD(Splited_set$dataTrain, alpha = 0.01, maxIter = 1000, seed = 123)
toc() # 2.5 sec

dim(grad_descent)

# Intercept term
intercept <- grad_descent[1]

term_weights_matrix_excl_intercept_term <- grad_descent[2:816]
term_weights_matrix_excl_intercept_term

term_weights_matrix_excl_intercept_term <- as.matrix(term_weights_matrix_excl_intercept_term)
dim(term_weights_matrix_excl_intercept_term) # 815 by 1
dim(Splited_set$dataTest[,-1]) # 36 by 815

# We need to transform both matrices to ensure conformable arguments
pred <- intercept + t(term_weights_matrix_excl_intercept_term) %*% t(Splited_set$dataTest[,-1])
dim(pred) # 1 by 36

# Step 6 : Validation using the Mean Absolute Error on test dataset

# Actual label we have assigned at the start of analysis
actual_label <- Splited_set$dataTest[,816]
actual_label

# Predicted label after we perform Gradient Descent

predicted_label <- pred
predicted_label

# Finding the residual error 
error <- (actual_label - predicted_label)
MAE <- mean(abs(error))
MAE # 1.6639

# Step 7 : Using another algorithm called Stochastic Gradient Descent (SGD)
tic()
SGD <- SGD(Splited_set$dataTrain, alpha = 0.01, maxIter = 1000, seed = 123)
toc() # 1.46 sec

dim(SGD)

# Intercept term
intercept <- SGD[1]

term_weights_matrix_excl_intercept_term <- SGD[2:816]
term_weights_matrix_excl_intercept_term
term_weights_matrix_excl_intercept_term <- as.matrix(term_weights_matrix_excl_intercept_term)
dim(term_weights_matrix_excl_intercept_term) # 815 by 1
dim(Splited_set$dataTest[,-1]) # 36 by 815


# We need to transform both matrices to ensure conformable arguments
pred <- intercept + t(term_weights_matrix_excl_intercept_term) %*% t(Splited_set$dataTest[,-1])
dim(pred) # 1 by 36

# Actual label we have assigned at the start of analysis
actual_label <- Splited_set$dataTest[,816]
actual_label

# Predicted label after we perform Gradient Descent

predicted_label <- pred
predicted_label

# Finding the residual error 
error <- (actual_label - predicted_label)
MAE <- mean(abs(error))
MAE # 1.6485


# Step 6 : Using another algorithm called Momentum Gradient Descent (MGD)

tic()
MGD <- MGD(Splited_set$dataTrain, alpha = 0.01, maxIter = 1000, momentum = 0.9, seed = 123)
toc() # 2.75 sec

dim(MGD)

# Intercept term
intercept <- MGD[1]

term_weights_matrix_excl_intercept_term <- MGD[2:816]
term_weights_matrix_excl_intercept_term
term_weights_matrix_excl_intercept_term <- as.matrix(term_weights_matrix_excl_intercept_term)
dim(term_weights_matrix_excl_intercept_term) # 815 by 1
dim(Splited_set$dataTest[,-1]) # 36 by 815

# We need to transform both matrices to ensure conformable arguments
pred <- intercept + t(term_weights_matrix_excl_intercept_term) %*% t(Splited_set$dataTest[,-1])
dim(pred) # 1 by 36

# Actual label we have assigned at the start of analysis
actual_label <- Splited_set$dataTest[,816]
actual_label

# Predicted label after we perform Gradient Descent

predicted_label <- pred
predicted_label

# Finding the residual error 
error <- (actual_label - predicted_label)
MAE <- mean(abs(error))
MAE # 1.6381

# MAE for GD is 1.6639 ( 2.5 sec), for SGD it's 1.6485 (1.46 sec)
#...and for MGD it's 1.6381 (2.56 sec) for alpha = 0.01 and max_iterations of 1000
