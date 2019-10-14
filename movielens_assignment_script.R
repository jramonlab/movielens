########################################
# Estimated full Run time :  15  minutes
start_time <- Sys.time()
########################################


################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#############################################################
#
# First, we need to split edx set for tuning/regularization purposes
#
test_index <- createDataPartition(y = edx$rating, times = 1, 
                                  p = 0.1, list = FALSE)
edx_train <- edx[-test_index,]
edx_test_temp <- edx[test_index,]
#
# Make sure userId and movieId in tets set are also in train set
#
edx_test <- edx_test_temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")
# Add rows removed from test set back into train set
#
removed <- anti_join(edx_test_temp, edx_test)
edx_train <- rbind(edx_train, removed)
rm(edx_test_temp, removed, test_index)
#
############################################################

# For future uses we avoid runing previous code
save(movielens, file  = "rda/movielens.rda")
save(edx, file        = "rda/edx.rda")
save(edx_test, file   = "rda/edx_test.rda")
save(edx_train, file  = "rda/edx_train.rda")
save(validation, file = "rda/validation.rda")

rm(dl, ratings, movies, temp, movielens)


# load("rda/movielens.rda")
load("rda/edx.rda")
load("rda/edx_test.rda")
load("rda/edx_train.rda")
load("rda/validation.rda")
load("rda/rmse_results.rda")
#############################################################

library(dplyr)
library(ggplot2)

options(pillar.sigfig = 5, pillar.subtle = FALSE, pillar.bold = TRUE)

#############################################################
# RMSE
RMSE <- function(true_ratings, predicted_ratings)
{
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
#############################################################

# Let´s train several models


# Average Model
#
mu_hat <- mean(edx$rating)
mu_hat

avg_rmse <- RMSE(validation$rating, mu_hat)
avg_rmse

rmse_results <- data_frame(METHOD = "Just the average", 
                           RMSE = avg_rmse)
rmse_results
#############################################################

# Adding movie effect to the model
#
mu <- mean(edx$rating) 
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Moniroring Movie variability
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

# Predictions  (Yui_hat  = mu + b_i)
predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)


model_1_rmse <- RMSE(validation$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(METHOD="Movie Effect Model",
                                     RMSE = model_1_rmse ))  
rmse_results


#############################################################
# Adding user effect to the model
#
# Monitoring the user variability
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# Calculating b_u
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predictions  ( Yui = mu + b_i + b_u )
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

# RMSE
model_2_rmse <- RMSE(validation$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(METHOD="Movie + User Effect Model",
                                     RMSE = model_2_rmse ))  
rmse_results


#############################################################
#
# HOw many movies with less than 50 ratings -> 5% of the movies
edx %>% count(title) %>% count(n < 5)
# Proportion of movies with less than 5 ratings
edx %>% 
  group_by(title) %>%
  summarize(n_ratings = n()) %>%
  tally(n_ratings < 5) %>%
  pull(n) / n_distinct(edx$movieId)
#
# Database that connects movieId to movie title:
# 
movie_titles <- movielens %>% 
  select(movieId, title) %>%
  distinct()

# How often are rated movies with extreme values of b_i
#
edx %>% count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  head(10)


#############################################################
# Regularization
#############################################################

#############################################################
# Regularization (Movie).
# It will help us penalize the movies with low 
# number of ratings which causes large variabiliy

###########################################################
# First, Optimizing lambda : penalization for low often rated movies
# 
lambdas <- seq(0, 5, 0.1)

mu <- mean(edx_train$rating)

just_the_sum <- 
  edx_train %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  
  predicted_ratings <- edx_test %>%
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  
  return(RMSE(edx_test$rating, predicted_ratings))
})
qplot(lambdas, rmses)  
best_lambda <- lambdas[which.min(rmses)]

#############################################################
# Now, Regularization (Movie).
# bi =  Sum (Yui - mu)/(n + lambda)
# for large n movies lambda is not significant
# lambda <- 1.6
lambda <- best_lambda

mu <- mean(edx$rating)
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

# Predictions (Yui_hat = mu + bi_reg + bu
predicted_ratings <- validation %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

model_3_rmse <- RMSE(validation$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(METHOD="Regularized Movie Effect Model",  
                                     RMSE = round(model_3_rmse,5)))
rmse_results 



###########################################################
# Conclusion about regularazing the movie ratings:
# Note that despite we have optimized lambda the RMSE is the same for both models (original 
# and regularized movie ratings).
# After analizing the number of movies wih less than 5 ratings we realize it is only 5% 
# movies. During the course while working with a smaller subset that proportion was of 70%, 
# in this case we did obserbed how the regularization improved hte RMSE.
###########################################################


#############################################################
# Regularization (User).
# It will help us penalize the movies with low 
# number of ratings which causes large variabiliy
#############################################################

# Optimizing Lambda (by using cross-validation on edx set)
# 
lambdas <- seq(3, 7, 0.1)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx_train$rating)
  
  b_i <- edx_train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  # Train set is use for tuning
  #
  predicted_ratings <- 
    edx_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(edx_test$rating, predicted_ratings))
})

qplot(lambdas, rmses)
best_lambda <- lambdas[which.min(rmses)]
best_lambda 
min(rmses)

# Now rmse is calculated with optimized lambda 
# RMSE over the validation set is calcutaed as follows:
l <- best_lambda # l = 4.6
mu <- mean(edx$rating)
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n() + l))

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n() + l))

# Validation set is used for obtaining final RMSE
#
predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

model_4_rmse <- RMSE(validation$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(METHOD="Regularized Movie + Regularized User Model",  
                                     RMSE = model_4_rmse))
rmse_results


# Why regularization does not improve rmse ?
edx %>% 
  group_by(userId) %>%
  summarize( n = n()) %>%
  arrange(n) %>%
  filter( n < 250) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 50, color = "black")

# Users with less than 20 ratings
low_raters <- edx %>% 
  group_by(userId) %>%
  summarize( n = n()) %>%
  filter(n < 20) %>% count()

p <- low_raters/n_distinct(edx$userId)
p
# 5 % of the raters have rated less than 20 movies
# On the course the proportion was of 10 per cent



#######################################################???
# Genre Effect
# # Predictions  ( Yui = mu + b_i + b_u  + b_g)

##################################################################
# optimizing lambda on edx set
#
lambdas <- seq(0, 6, 0.25)
rmses_extended <- sapply(lambdas, function(l){
  
  mu <- mean(edx_train$rating)
  
  b_i <- edx_train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+ l))
  
  b_u <- edx_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+ l))
  
  b_g <- edx_train %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_u - b_i - mu)/(n() + l))
  
  # Train set is use for tuning
  #
  predicted_ratings <- 
    edx_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)
  
  return(RMSE(edx_test$rating, predicted_ratings))
})

qplot(lambdas, rmses_extended)
best_lambda <- lambdas[which.min(rmses_extended)]
best_lambda
min(rmses)
l <- best_lambda 

# Model based on regulaized movie + user + genre
# Now computed wihth optimized lambda
#
mu <- mean(edx$rating)

b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + l))

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu) / (n() + l))

b_g <- edx %>% 
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_u - b_i - mu) / (n() + l))


# Predictions  ( Yui = mu + b_i + b_u  + b_g) over validation set
predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

model_5_rmse <- RMSE(validation$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results, data_frame(METHOD = "Regularized Movie + Regularized User Effect Model + Reg. Genre",
                                                   RMSE = model_5_rmse))
rmse_results
save(rmse_results, file =  "rda/rmse_results.rda")

cat("Run time : ", Sys.time() - start_time , " minutes")





##############################################

# EDA on genre and time
 
###############################################

# Reducing genres groups from 787 to 20 GROUPS
library(stringr)

# list original genres
genres <-edx %>% group_by(genres) %>% summarize(n = n()) %>%
  arrange(n)

head(edx, 10)  %>% 
  str_sub(string = .$genres, 
          start = 1, 
          end = str_locate(string = .$genres, pattern = "\\|") )

# Extract first genre as the manin one
#
main_genre  <- edx %>% pull(genres) %>%
    str_extract(pattern = "[A-z]{1,11}") 


# list main genre (20 groups)
# Let´s observe how ratings are distributted per genre
genres <-edx %>% mutate(main_genre = main_genre) %>%
  group_by(main_genre) %>% 
  summarize(n = n()) %>%
  arrange(n)

# Let´s observe how movies are distributted per main genre
edx %>% mutate(main_genre = main_genre) %>%
  group_by(main_genre) %>% 
  summarize(n = n_distinct(movieId)) %>%
  arrange(n)


library(ggrepel)
library(ggthemes)

edx %>% mutate(main_genre = main_genre) %>%
  group_by(main_genre) %>% 
  summarize(n_ratings = n(), n_movies = n_distinct(movieId), rate = n_ratings/n_movies, avg_rating = mean(rating)) %>%
  arrange(avg_rating) %>% 
  ggplot(aes(x = rate, y = avg_rating, label = main_genre)) +
  geom_point(size = 2.5) + 
  geom_text_repel() +
  geom_smooth(method = "lm") + 
  xlab("Number of ratings / number of movies classified into each main genre") + 
  ylab("AVG RATING") +
  ggtitle("Relation between Rating frequency and avg Rating obtained by Main Genre")
  theme_minimal()

ggsave("figs/Main_genre_classification.png", width = 10, height = 10)

# No strong correlation between number of ratings per genre and the rating obtained per genre. 
# Nevertheless, it seems the highest ratings are obtained for thos movies whoch less rated

# Let´s observe if some tree model based on the main genre ? 

# Let´s observe TIME SPAN while a movie is rated
library(lubridate)

edx %>% mutate(date = as_datetime(timestamp)) %>%
  mutate(main_genre = main_genre) %>%
  group_by(movieId) %>%
  mutate(time_span = (max(date) - min(date))/(60*60*24)) %>%
  ungroup() %>%
  group_by(main_genre) %>%
  summarize(avg_time_span = mean(time_span), avg_rating = mean(rating)) %>%
  
  ggplot(aes(avg_time_span, avg_rating, label = main_genre)) +
  geom_point(size = 2.5) + 
  geom_text_repel() +
  geom_smooth(method = "lm") + 
  xlab("Days being rated") + 
  ylab("AVG RATING") +
  ggtitle("Relation between Rating time span and avg Rating obtained by Main Genre") +
  theme_minimal()

ggsave("figs/main_genre_time_classification.png", width = 5, height = 5)
  
















###################################################################


# EDA on Movielens. Plots for report


###################################################################

library(dplyr)
library(ggplot2)
library(ggthemes)

load("rda/movielens.rda")

mu <- mean(movielens$rating)

# Monitoring the movie variability
movielens %>% 
  group_by(movieId) %>% 
  #summarize(b_i = mean(rating)) %>%
  summarize(b_i = mean(rating - mu)) %>% 
  ggplot(aes(b_i)) + 
  geom_histogram(bins = 30, fill = "lightblue", color = "grey") +
  ggtitle("Movies [from movielens]") +
  xlab("b_i [ Rating grades - mu ]") + 
  ylab("Number of movies") +
  theme_minimal()

ggsave("figs/b_i_movies.png")

# Monitoring the user variability
movielens %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, fill = "lightgreen", color = "grey") +
  ggtitle("Users [ from movielens ]") +
  xlab("b_u [ Rating grades - mu ]") + 
  ylab("Number of users") +
  theme_minimal()

ggsave("figs/b_u_users.png")

# Monitoring the number of ratings
movielens %>% group_by(movieId) %>% summarize(n = n()) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, fill = "lightblue", color = "grey", show.legend = TRUE) + 
  scale_x_log10() + 
  ggtitle("Movies [ from movielens ]") +
  xlab("Number of ratings (log10)") + 
  ylab("Number of movies") +
  theme_minimal()

ggsave("figs/movie_ratings.png", width = 5, height = 5)


movielens %>% group_by(userId) %>% summarize(n = n()) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30 , fill = "lightgreen", color = "grey") + 
  scale_x_log10() + 
  ggtitle("Users [from movielens where n < 1000]") +
  xlab("Number of ratings (log10)") + 
  ylab("Number of users")+
  theme_minimal()

ggsave("figs/user_ratings.png", width = 5, height = 5)

## REGULARIZATION

# Why regularization does not improve rmse ?
movielens %>% 
  group_by(movieId) %>%
  summarize( n = n()) %>%
  arrange(n) %>%
  filter( n < 250) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black")

# Users with less than 30 ratings
#
low_raters <- movielens %>% 
  group_by(userId) %>%
  summarize( n = n()) %>%
  filter(n < 30) %>% count()

p <- low_raters/n_distinct(movielens$userId)
p
# 20 % of the raters have rated less than 30 movies
# On the course the proportion was of 10 per cent


# Movies with less than 5 ratings (4.5%)
#
low_rated_movies <- movielens %>% 
  group_by(movieId) %>%
  summarize( n = n()) %>%
  filter(n < 5) %>% count()

p <- low_rated_movies/n_distinct(movielens$movieId)
p
# This proportion for movielens-100k was about 70%



cat("Run time : ", Sys.time() - start_time , " minutes")



























