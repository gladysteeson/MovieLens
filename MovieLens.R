################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

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
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
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

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# tidy format of 'edx' data
edx %>% tibble()

# Convert timestamp to date format using lubridate function, add columns for rating_year and release_year
edx_mut <- edx %>% mutate(timestamp = as_datetime(timestamp), rating_year = year(as_datetime(timestamp)), release_year = as.numeric(str_sub(title,-5,-2)))
head(edx_mut)

# structure of our new edx dataset
str(edx_mut)

# number of unique users and movies 
edx_mut %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId)) 

# sparse matrix for sampled 100 unique userId and moveiId 
users <- sample(unique(edx_mut$userId), 100) 
rafalib::mypar() 
edx_mut %>% filter(userId %in% users) %>%  
  select(userId, movieId, rating) %>% 
  mutate(rating = 3) %>% 
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>%  
  as.matrix() %>% t(.) %>% 
  image(1:100, 1:100,. , xlab="Movies", ylab="Users") 
abline(h=0:100+0.5, v=0:100+0.5, col = "grey") 


# distribution of movie ratings 
edx_mut %>%  
  dplyr::count(movieId) %>%  
  ggplot(aes(n)) +  
  geom_histogram(bins = 30, color = "black") +  
  scale_x_log10() +  
  ggtitle("Movies") 

# distribution of users 
edx_mut %>% 
  dplyr::count(userId) %>%  
  ggplot(aes(n)) +  
  geom_histogram(bins = 30, color = "black") +  
  scale_x_log10() + 
  ggtitle("Users") 


# top 10 movies with the most ratings per year since its release
years_rated_max <- max(edx_mut$rating_year) + 1
edx_mut %>% group_by(movieId) %>%
  summarize(n = n(), title = title[1], years = years_rated_max - first(release_year), avg_rating = mean(rating)) %>%
  mutate(rate = n/years) %>%
  top_n(10, rate) %>%
  arrange(desc(rate))

# Define RMSE function for vectors of ratings and their corresponding predictors 
RMSE <- function(true_ratings, predicted_ratings){ 
  sqrt(mean((true_ratings - predicted_ratings)^2)) 
} 

# First Model: same rating for all movies using average rating of all movies

# calculate average rating of all movies 
mu_hat <- mean(edx_mut$rating) 
mu_hat 
# calculate rmse  
naive_rmse <- RMSE(validation$rating, mu_hat) 
naive_rmse 
# add rmse results in a table 
rmse_results <- data.frame(method = "Simple Average Rating Model", RMSE = naive_rmse) 
rmse_results %>% knitr::kable() 


# Second Model: movie effect on average rating

# calculate movie bias 'b_i' for all movies 
mu <- mean(edx_mut$rating) 
movie_avgs <- edx_mut %>%  
  group_by(movieId) %>%  
  summarize(b_i = mean(rating - mu)) 
# plot the movie 'bias' 
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black")) 
# calculate predictions using movie effect 
predicted_ratings <- mu + validation %>%  
  left_join(movie_avgs, by='movieId') %>% 
  pull(b_i) 
# calculate rmse 
model_1_rmse <- RMSE(predicted_ratings, validation$rating) 
# add rmse results in to the table 
rmse_results <- bind_rows(rmse_results, 
                          data_frame(method="Movie Effects Model",   
                                     RMSE = model_1_rmse)) 
rmse_results %>% knitr::kable() 


# Third Model: combining user effect and movie effect

# plot of avg rating for users that've rated over 100 movies 
edx_mut %>% group_by(userId) %>%  
  summarize(b_u = mean(rating)) %>%  
  filter(n() >= 100) %>%  
  ggplot(aes(b_u)) +  
  geom_histogram(bins = 30, color = "black") 
# calculate user bias 'b_u' for all users 
user_avgs <- edx_mut %>%  
  left_join(movie_avgs, by='movieId') %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i)) 
# calculate predictions using user effects along with movie effects
predicted_ratings <- validation %>%  
  left_join(movie_avgs, by='movieId') %>% 
  left_join(user_avgs, by='userId') %>% 
  mutate(pred = mu + b_i + b_u) %>% 
  pull(pred) 
# calculate rmse 
model_2_rmse <- RMSE(predicted_ratings, validation$rating) 
rmse_results <- bind_rows(rmse_results, 
                          data_frame(method="Movie + User Effects Model",   
                                     RMSE = model_2_rmse)) 
rmse_results %>% knitr::kable() 


# Fourth Model: regularizing movie + user effect 

# top 10 and bottom 10 movies according to our estimates
movie_titles <- edx_mut %>% 
  select(movieId, title) %>%
  distinct()
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10)  %>% 
  pull(title)
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  slice(1:10)  %>% 
  pull(title)

# examine how often the top movies are rated
edx_mut %>% count(movieId) %>% 
  left_join(movie_avgs, by="movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10) %>% 
  pull(n)

# examine how often the worst movies are rated
edx_mut %>% count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  slice(1:10) %>% 
  pull(n)

# choosing the penalty term lambda 

# Create additional Partition from the edx_mut test set into edx_mut_test and edx_mut_val to obtain lambda
test_index <- createDataPartition(y = edx_mut$rating, times = 1, p = 0.1, list = FALSE)
edx_mut_test <- edx_mut[-test_index,]
edx_mut_temp <- edx_mut[test_index,]

# Make sure userId and movieId in validation set are also in edx_mut_test set
edx_mut_val <- edx_mut_temp %>% 
  semi_join(edx_mut_test, by = "movieId") %>%
  semi_join(edx_mut_test, by = "userId")


# Add rows removed from validation set back into edx_mut_test set
removed <- anti_join(edx_mut_temp, edx_mut_val)
edx_mut_test <- rbind(edx_mut_test, removed)

# find optimal lambda

lambdas <- seq(0, 10, 0.5) 


rmses <- sapply(lambdas, function(l){ 
  
  mu <- mean(edx_mut_test$rating) 
  
  b_i <- edx_mut_test %>%  
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l)) 
  
  b_u <- edx_mut_test %>%  
    left_join(b_i, by="movieId") %>% 
    group_by(userId) %>% 
    summarize(b_u = sum(rating - b_i - mu)/(n()+l)) 
  
  predicted_ratings <-  
    edx_mut_val %>%  
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    mutate(pred = mu + b_i + b_u) %>% 
    pull(pred) 
  
  return(RMSE(predicted_ratings, edx_mut_val$rating)) 
}) 


qplot(lambdas, rmses)  


# find lambda that minimizes rmse 
lambda <- lambdas[which.min(rmses)] 
lambda 

# using the optimal lambda, run the model with the original edx_mut train and validation set
mu <- mean(edx_mut$rating) 

b_i <- edx_mut %>%  
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda)) 

b_u <- edx_mut %>%  
  left_join(b_i, by="movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda)) 

predicted_ratings <-  
  validation %>%  
  left_join(b_i, by = "movieId") %>% 
  left_join(b_u, by = "userId") %>% 
  mutate(pred = mu + b_i + b_u) %>% 
  pull(pred) 


# calculate rmse
RMSE_Regularized <- RMSE(predicted_ratings, validation$rating)

rmse_results <- bind_rows(rmse_results, 
                          data_frame(method="Regularized Movie + User Effect Model",   
                                     RMSE = RMSE_Regularized))
rmse_results %>% knitr::kable()

