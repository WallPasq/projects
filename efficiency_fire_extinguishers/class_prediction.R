# Project developed during the course Big Data Analytics with R and Microsoft Azure Machine Learning,
# from the Data Science Academy: https://www.datascienceacademy.com.br/course/analise-de-dados-com-r

#### Context ####

# The company Extinflames works with the commercialization of fire extinguishers and, in order to bring
# more safety to its customers and differentiate itself in the market, it wants to bring a test capable
# of identifying the operation of an extinguisher through computer simulations.


#### Business question ####

# Would it be possible to use computer simulations to predict the operation of a fire extinguisher, thus
# adding an additional layer of safety to my customers' operations?


#### How to solve it? ####

# Using a Machine Learning model of classification, with the R language.


#### Definition of done ####

# The model must have an Accuracy and Area Under the Curve (AUC) of at least 0.95.


#### About data ####

# Reference
# KOKLU M., TASPINAR Y.S., (2021). Determining the Extinguishing Status of Fuel Flames With Sound Wave
# by Machine Learning Methods. IEEE Access, 9, pp.86207-86216, Doi: 10.1109/ACCESS.2021.3088612.

# Summary
# The dataset of the study was obtained as a result of the extinguishing tests of four different fuel
# flames with a sound wave extinguishing system. The sound wave fire-extinguishing system consists of
# 4 subwoofers with a total power of 4,000 Watt placed in the collimator cabinet. There are two
# amplifiers that enable the sound come to these subwoofers as boosted. Power supply that powers the
# system and filter circuit ensuring that the sound frequencies are properly transmitted to the system
# is located within the control unit. While computer is used as frequency source, anemometer was used
# to measure the airflow resulted from sound waves during the extinguishing phase of the flame, and a
# decibel meter to measure the sound intensity. An infrared thermometer was used to measure the
# temperature of the flame and the fuel can, and a camera is installed to detect the extinction time of
# the flame. A total of 17,442 tests were conducted with this experimental setup. The experiments are
# planned as follows:
#
# 1) Three different liquid fuels and LPG fuel were used to create the flame.
# 2) Five different sizes of liquid fuel cans are used to achieve different size of flames.
# 3) Half and full gas adjustment is used for LPG fuel.
# 4) While carrying out each experiment, the fuel container, at 10cm distance, was moved forward up to
#    190cm by increasing the distance by 10 cmeach time.
# 5) Along with the fuel container, anemometer and decibel meter were moved forward in the same
#    dimensions.
# 6) Fire extinguishing experiments was conducted with 54 different frequency sound waves at each
#    distance and flame size.
#
# Throughout the flame extinguishing experiments, the data obtained from each measurement device was
# recorded and a dataset was created. The dataset includes the features of fuel container size
# representing the flame size, fuel type, frequency, decibel, distance, airflow and flame extinction.
# Accordingly, 6 input features and 1 output feature will be used in models.

# Data source:
# https://www.muratkoklu.com/datasets/vtdhnd07.php


#### Data dictionary ####

# |--------------|------------------------------------------------------------|
# | Name         | Definition                                                 |
# |--------------|------------------------------------------------------------|
# | size         | Flame size in categorical variable:                        |
# |              | 1 = 7cm (does not exist for LPG fuel);                     |
# |              | 2 = 12cm (does not exist for LPG fuel);                    |
# |              | 3 = 14cm (does not exist for LPG fuel);                    |
# |              | 4 = 16cm (does not exist for LPG fuel);                    |
# |              | 5 = 20cm (does not exist for LPG fuel);                    |
# |              | 6 = Half throttle setting (on LPG fuel only);              |
# |              | 7 = Full throttle setting (on LPG fuel only).              |
# |--------------|------------------------------------------------------------|
# | fuel         | The type of fuel used to produce the flame, which may be:  |
# |              | Gasoline, Kerosene, Thinner or LPG.                        |
# |--------------|------------------------------------------------------------|
# | distance     | Distance of flame to collimator output.                    |
# |              | Continuous variable between 10cm and 190cm.                |
# |--------------|------------------------------------------------------------|
# | decibel      | Sound pressure level.                                      |
# |              | Continuous variable between 72dB and 113dB.                |
# |--------------|------------------------------------------------------------|
# | airflow      | Airflow created by the sound wave.                         |
# |              | Continuous variable between 0m/s and 17m/s.                |
# |--------------|------------------------------------------------------------|
# | frequency    | Low frequency range.                                       |
# |              | Continuous variable between 1Hz and 75Hz.                  |
# |--------------|------------------------------------------------------------|
# | status       | The target variable:                                       |
# |              | 0 = non-extinction state;                                  |
# |              | 1 = extiction state.                                       |
# |--------------|------------------------------------------------------------|


#### Loading, cleaning and exploratory data analysis ####
library(readxl)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(gridExtra)
library(corrgram)
library(caret)
library(class)
library(neuralnet)
library(pROC)

# Checks which are the sheets of the Excel file.
file <- 'dataset.xlsx'
sheets <- excel_sheets(file)
dbs <- list()

for (s in sheets) {
  dbs[[s]] <- read_excel(file, s)
}

View(dbs)

# Sheets 2 and 3 have already been duly mentioned in the comments above.
# Let's work only with sheet 1.

db <- dbs[[1]]
rm(dbs)
rm(sheets)
rm(file)
rm(s)

View(db)
dim(db) # There are 17,442 observations and 7 variables.
str(db)
names(db) <- tolower(names(db))
names(db)[grep('desibel', names(db))] <- 'decibel'

colSums(is.na(db)) # There are no missing values in the dataset.

# Let's transform some columns to explore their data.
db$fuel <- factor(db$fuel)
db$status <- factor(db$status)
db$status_desc <- db$status
levels(db$status_desc) <- c('non-extinction', 'extinction')

# Let's check the distribution of the target variable.
db %>% 
  mutate(total = n()) %>% 
  group_by(status, status_desc) %>% 
  summarise(amount = n(),
            perc = amount / mean(total))

# The distribution is pretty close, almost 50/50. Let's check how it is by type of fuel.
db %>% 
  group_by(fuel) %>% 
  mutate(total = n()) %>% 
  ungroup() %>% 
  group_by(fuel, status_desc) %>% 
  summarise(amount = n(),
            perc = amount / mean(total)) %>% 
  arrange(fuel, status_desc)

# The distribution is also very close between each type of fuel.
# Let's check the distribution of the other columns with the target variable.
db <- select(db, size, distance, decibel, airflow, frequency, fuel, status, status_desc)

lapply(names(db)[1:5], function(y) {
  ggplot(db[, c('status_desc', y)], aes_string(x='status_desc', y=y)) +
    geom_boxplot() +
    labs(title=paste('Distribution of', y, 'by status'))
})

# We have the following:
# Size: Apparently, the fire is more likely to be put out if the flame is lower, given
#       that most of the data are below the median for non-extinction, and above the
#       median for extinction. Let's check this out in another chart.
db %>%
  select(status_desc, size) %>%
  mutate(size=factor(size)) %>%
  group_by(size) %>% 
  mutate(total=n()) %>% 
  ungroup() %>% 
  group_by(status_desc, size) %>%
  summarise(amount=n(),
            perc=amount/mean(total)) %>% 
  ungroup() %>% 
  ggplot(aes(x=size, y=perc, fill=status_desc)) +
    geom_bar(position='dodge', stat='identity') +
    geom_text(aes(label=scales::percent(perc)),
              position=position_dodge(width=0.9),
              vjust=-0.5) +
    labs(title='Flame size distribution by status') +
    theme(axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank())

# Considering only ratios close to or greater than 60/40, there are more extinguished flames
# of 7cm, and more unextinguished flames of 16cm and 20cm. In the case of LPG fuel, there are
# more extinguished flames produced by the half throttle setting. The remaining cases present
# a very similar relationship between status, almost 50/50.

# Distance: Looking at the chart, it is clear that the greater the distance, the less likely
#           the flame is to extinguish and vice versa. Let's check the distribution of distances
#           by status with a histogram.
ggplot(db, aes(x=distance)) +
  geom_histogram() +
  facet_wrap(~status_desc, nrow=2) +
  labs(title='Distance histogram by status.')

# The histogram proves the vision we had with the boxplot: the greater the distance, the less
# likely the flame is to be extinguished.

# Decibel: It seems that as the decibel increases, the probability of the flame going out
#          increases slightly. Let's check this out on a histogram as well.
ggplot(db, aes(x=decibel)) +
  geom_histogram() +
  facet_wrap(~status_desc, nrow=2) +
  labs(title='Decibel histogram by status.')

# There seems to be a small correlation between the increase in the decibel and whether or not
# the flame is extinguished. Let's check with the Wilcoxon/Mann-Whitney test.
# H0: The two samples belong to the same measure of central tendency;
# Ha: Samples are centered at different points.
wilcox.test(decibel ~ status, db)

# The p-value is less than 0.05. Therefore, we reject the null hypothesis, and take it as true
# that they belong to different measures of central tendency. Based on the chart, we assume
# that increasing the decibel helps to extinguish the flame.

# Let's also see if there is any correlation between distance and decibel, by status.
ggplot(db, aes(x=distance, y=decibel)) +
  geom_point() +
  geom_smooth(method=lm) +
  stat_cor(p.accuracy=0.001, r.accuracy=0.01) +
  facet_wrap(~status_desc, ncol=1) +
  labs(title='Scatter plot of distance by decibel, segmented by status.')

# The correlation is very weak between the variables in both status: non-extinction and
# extinction. Let's split the histogram by status and fuel.
ggplot(db, aes(x=decibel)) +
  geom_histogram() +
  facet_wrap(~fuel + status_desc, ncol=2, nrow=4) +
  labs(title='Decibel histogram by status and fuel.')

# There seem to be two distributions of the decibel variable: one between 70 and 100 and
# the other between 100 and 110. However, we still haven't found a variable that divides
# it into the two distributions. Perhaps it is better to transform this variable into four
# classes: <=85 (moderate), >85 and <=95 (moderate high), >95 and <=100 (high), >100 (very high).
db %>% 
  mutate(decibel_CAT=cut(decibel,
                         breaks=c(-Inf, 85, 95, 100, Inf),
                         labels=c('moderate', 'moderate high', 'high', 'very high'))) %>% 
  group_by(decibel_CAT) %>% 
  mutate(total=n()) %>% 
  ungroup() %>% 
  group_by(status_desc, decibel_CAT) %>%
  summarise(amount=n(),
            perc=amount/mean(total)) %>% 
  ungroup() %>% 
  ggplot(aes(x=decibel_CAT, y=perc, fill=status_desc)) +
  geom_bar(position='dodge', stat='identity') +
  geom_text(aes(label=scales::percent(perc)),
            position=position_dodge(width=0.9),
            vjust=-0.5) +
  labs(title='Categorical decibel distribution by status') +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank())

# If the decibel is less than or equal to 85 or between 95 and 100, the probability of the flame
# extinguishing is very low. If it is greater than 100, the probability of the flame extinguishing
# is greater. Let's save this information, and later we will decide if we are going to use it or not
# in the construction of the model.

# Airflow: Observing the boxplot, it is clear that as the airflow increases, the probability
#          of the flame being extinguished also increases. However, there are many outliers
#          to both status. Let's check this out with a histogram as well.
ggplot(db, aes(x=airflow)) +
  geom_histogram() +
  facet_wrap(~fuel + status_desc, ncol=2, nrow=4) +
  labs(title='Airflow histogram by status and fuel.')

# There is a behavior pattern for airflow in both status and fuel. When the airflow is 0,
# practically no flames are extinguished, and as the airflow increases, so does the probability
# that the flame will be extinguished.
db %>% 
  mutate(airflow_CAT=cut(airflow,
                         breaks=seq(0, 18, 3.6),
                         labels=c('very weak', 'weak', 'medium',
                                  'strong', 'very strong'),
                         include.lowest=TRUE)) %>% 
  group_by(airflow_CAT) %>%
  mutate(total=n()) %>%
  ungroup() %>%
  group_by(status_desc, airflow_CAT) %>%
  summarise(amount=n(),
            perc=amount/mean(total)) %>%
  ungroup() %>%
  ggplot(aes(x=airflow_CAT, y=perc, fill=status_desc)) +
    geom_bar(position='dodge', stat='identity') +
    geom_text(aes(label=scales::percent(perc)),
              position=position_dodge(width=0.9),
              vjust=-0.5) +
    labs(title='Categorical airflow distribution by status') +
    theme(axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank())

# Perhaps it is better to transform this variable into a categorical one, with five categories.
# Let's save this information to decide on building the model.

# Frequency: Smaller Hz bands seem to be more effective in extinguishing the flame than larger
#            bands. However, they are very close. Let's also check it out on a histogram.
ggplot(db, aes(x=frequency)) +
  geom_histogram() +
  facet_wrap(~fuel + status_desc, ncol=2, nrow=4) +
  labs(title='Frequency histogram by status and fuel.')

# The distributions are very similar. Let's check with the Wilcoxon/Mann-Whitney test if this
# variable has a different tendency by status.
# H0: The two samples belong to the same measure of central tendency;
# Ha: The samples are centered at different points.
wilcox.test(frequency ~ status, db)

# The p-value is less than 0.05. Therefore, we reject the null hypothesis, and take it as true
# that they belong to different measures of central tendency. Based on the chart, we assume
# that decreasing the frequency helps to extinguish the flame.

# Let's analyze the correlations between numerical variables.
corrgram(db %>% select(where(is.numeric)),
         lower.panel=panel.pts,
         diag=panel.density,
         upper.panel=panel.cor)

# We have a negative correlation between distance and airflow (-0.71). In other cases, the
# correlation is very small.

#### Preprocessing ####

# Based on the exploratory analysis, let's do the following:
# 1) Create a categorical column for Size;
# 2) Create a categorical column for Decibel;
# 3) Create a categorical column for Airflow;
# 4) Divide the data into train and test;
# 5) Normalize numerical values.

db <- db %>%
  mutate(size_CAT=factor(size),
         decibel_CAT=cut(decibel,
                         breaks=c(-Inf, 85, 95, 100, Inf),
                         labels=c('moderate', 'moderate high', 'high', 'very high')),
         airflow_CAT=cut(airflow,
                         breaks=seq(0, 18, 3.6),
                         labels=c('very weak', 'weak', 'medium',
                                  'strong', 'very strong'),
                         include.lowest=TRUE))

set.seed(42)
db['TrainTest'] <- sapply(runif(nrow(db)), function(x) if (x < 0.3) 0 else 1)

X_train <- db %>% filter(TrainTest == 1) %>% select(-status, -status_desc, -TrainTest)
y_train <- db %>% filter(TrainTest == 1) %>% select(status, status_desc)
X_test <- db %>% filter(TrainTest == 0) %>% select(-status, -status_desc, -TrainTest)
y_test <- db %>% filter(TrainTest == 0) %>% select(status, status_desc)

print(nrow(X_train) / nrow(db)) # 69.60% of the dataset
print(nrow(X_test) / nrow(db)) # 30.40% of the dataset

# Normalizing the values avoiding leakage of data from test to train.
my_norm <- function(x) {
  var_min=min(x)
  var_max=max(x)
  var_func=function(y) return((y - var_min) / (var_max - var_min))
  inv_func=function(z) return(z * (var_max - var_min) + var_min)
  return(list(
      min=var_min,
      max=var_max,
      norm=var_func,
      unnorm=inv_func
  ))
}

norms <- list()

for (i in X_train %>% select(where(is.numeric)) %>% names()) {
  norms[[i]] <- my_norm(X_train[, i])
  X_train[paste(i, '_NORM', sep='')] <- norms[[i]]$norm(X_train[, i])
  X_test[paste(i, '_NORM', sep='')] <- norms[[i]]$norm(X_test[, i])
}

View(X_train)
View(X_test)
View(y_train)
View(y_test)
View(norms)

#### Predictive models ####

# Let's start with a very simple model, using only the normalized values.
model_knn_v1 <- knn(train=X_train[, grep('NORM', names(X_train))],
                    test=X_test[, grep('NORM', names(X_test))],
                    cl=y_train$status,
                    k=21, prob=TRUE)

confusionMatrix(data=model_knn_v1, reference=y_test$status)

# This model has an accuracy of 0.9353.
# Let's check the AUC of this model.
roc_knn_v1 <- roc(y_test$status, attr(model_knn_v1, 'prob'), percent=TRUE)
auc_knn_v1 <- function(x=roc_knn_v1) {
  plot(roc_knn_v1, print.auc=TRUE, legacy.axes=TRUE, col='blue', 
       xlab="False Positive Rate", ylab="True Positive Rate")
}

par(pty='s')
auc_knn_v1()
roc_knn_v1$auc

# This model has an AUC of 0.5132.
# Our first model performed well on accuracy but not on the AUC curve. Let's try
# to improve performance by using fuel type as a dummy variable.
for (i in levels(X_train$fuel)) {
  if (i != 'lpg') {
    X_train[paste('is_', i, sep='')] = sapply(X_train$fuel, function(x) ifelse(x==i, 1, 0))
    X_test[paste('is_', i, sep='')] = sapply(X_test$fuel, function(x) ifelse(x==i, 1, 0))
  }
}

View(X_train)
View(X_test)

model_knn_v2 <- knn(train=X_train[, grep('NORM|is_', names(X_train))],
                    test=X_test[, grep('NORM|is_', names(X_test))],
                    cl=y_train$status,
                    k=21, prob=TRUE)

confusionMatrix(data=model_knn_v2, reference=y_test$status)

# This model has an accuracy of 0.9598.
# Let's check the AUC of this model.
roc_knn_v2 <- roc(y_test$status, attr(model_knn_v2, 'prob'), percent=TRUE)
auc_knn_v2 <- function(x=roc_knn_v2) {
  plot(roc_knn_v2, print.auc=TRUE, legacy.axes=TRUE, col='green',
       print.auc.y=40, add=TRUE)
}

par(pty='s')
auc_knn_v1()
auc_knn_v2()
legend('bottomright', legend=c('without_fuel', 'with_fuel'),col=c('blue', 'green'), lwd=2)
roc_knn_v2$auc

# This model has an AUC of 0.5092.
# Model accuracy improved, but the AUC curve decreased. Let's try using another ML model.

formula_v1 <- as.formula(paste('status ~',
                         paste(names(X_train[, grep('NORM|is_', names(X_train))]),
                               collapse=' + ')))

model_neural_v1 <- neuralnet(formula_v1, data=cbind(X_train, y_train), hidden=c(5,3),
                             linear.output=FALSE, lifesign='full', threshold=0.5)

prev_neural_v1 <- data.frame(predict(model_neural_v1, X_test))
names(prev_neural_v1) <- c('neural_0', 'neural_1')
prev_neural_v1$prev_n <- max.col(prev_neural_v1)
prev_neural_v1 <- prev_neural_v1 %>% 
  mutate(prev=factor(levels(y_test$status)[prev_n]),
         prev_desc=factor(levels(y_test$status_desc)[prev_n])) %>% 
  select(-prev_n)

softmax <-  function(x) {
  return(exp(x) / rowSums(exp(x)))
}

probs_neural_v1 <- softmax(prev_neural_v1[, c(1, 2)])
names(probs_neural_v1) <- c('prob_0', 'prob_1')
prev_neural_v1 <- cbind(prev_neural_v1, probs_neural_v1) %>% 
  select(neural_0, neural_1, prob_0, prob_1, prev, prev_desc)
rm(probs_neural_v1)
View(prev_neural_v1)

# Let's calculate the model metrics.
confusionMatrix(data=prev_neural_v1$prev, reference=y_test$status)

# This model has an accuracy of 0.9528.
# Let's check the AUC of this model.
roc_knn_v3 <- roc(y_test$status, prev_neural_v1$prob_1, percent=TRUE)
auc_knn_v3 <- function(x=roc_knn_v3) {
  plot(roc_knn_v3, print.auc=TRUE, legacy.axes=TRUE, col='orange',
       print.auc.y=80, add=TRUE)
}

par(pty='s')
auc_knn_v1()
auc_knn_v2()
auc_knn_v3()
legend('bottomright', legend=c('knn_without_fuel', 'knn_with_fuel', 'neuralnet'),
       col=c('blue', 'green', 'orange'), lwd=2)
roc_knn_v3$auc

# This model has an AUC of 0.9909. Let's get on with it!
final_model <- model_neural_v1
rm(softmax)
rm(my_norm)
rm(model_knn_v1)
rm(model_knn_v2)
rm(i)
rm(formula_v1)
rm(roc_knn_v1)
rm(roc_knn_v2)
rm(roc_knn_v3)
rm(auc_knn_v1)
rm(auc_knn_v2)
rm(auc_knn_v3)
rm(prev_neural_v1)
rm(model_neural_v1)

# Function to make predictions with this model
status_prediction <- function(x, model=final_model) {
  
  # If you retrain the model, modify these mean and standard deviation values according to the values 
  # of the new training base.
  
  x$size_NORM <- sapply(x$size, function(y) (y - 1)/(7 - 1))
  x$distance_NORM <- sapply(x$distance, function(y) (y - 10)/(190 - 10))
  x$decibel_NORM <- sapply(x$decibel, function(y) (y - 72)/(113 - 72))
  x$airflow_NORM <- sapply(x$airflow, function(y) (y - 0) / (17 - 0))
  x$frequency_NORM <- sapply(x$frequency, function(y) (y - 1) / (75 - 1))
  x$is_gasoline <- sapply(x$fuel, function(y) ifelse(y=='gasoline', 1, 0))
  x$is_kerosene <- sapply(x$fuel, function(y) ifelse(y=='kerosene', 1, 0))
  x$is_thinner <- sapply(x$fuel, function(y) ifelse(y=='thinner', 1, 0))
  
  softmax <- function(z) {
    exp(z) / rowSums(exp(z))
  }
  
  pred <- predict(model, x)
  proba <- softmax(pred)
  pred <- data.frame(pred)
  names(pred) <- c('neural_0', 'neural_1')
  pred$prev_n <- max.col(pred)
  proba <- data.frame(proba)
  names(proba) <- c('proba_0', 'proba_1')
  
  return(
    cbind(pred, proba) %>% 
      mutate(prev=factor(c(0, 1)[prev_n]),
             prev_desc=factor(c('non-extinction', 'extinction')[prev_n])) %>% 
      select(-prev_n)
  )
}

db_test_predict <- db %>% select(size, distance, decibel, airflow, frequency, fuel, status)
X_test_predict <- db_test_predict[1:50, -7]
y_test_predict <- db_test_predict[1:50, 7]

prev_test <- status_prediction(X_test_predict)
View(prev_test)
confusionMatrix(data=prev_test$prev, reference=y_test_predict$status)

# The function works!
rm(db_test_predict)
rm(X_test_predict)
rm(y_test_predict)
rm(prev_test)
rm(norms)
rm(X_train)
rm(y_train)
rm(X_test)
rm(y_test)

#### Explaining the model ####

# This model has an accuracy of 95.28% and an AUC of 99.09%. It makes predictions about whether
# or not a fire extinguisher with a sound wave extinguishing system will put out a flame, based
# on six criteria:

# 1) Flame Size:
#   1.1) 1 = 7cm (does not exist for LPG fuel);
#   1.2) 2 = 12cm (does not exist for LPG fuel);
#   1.3) 3 = 14cm (does not exist for LPG fuel);
#   1.4) 4 = 16cm (does not exist for LPG fuel)
#   1.5) 5 = 20cm (does not exist for LPG fuel);
#   1.6) 6 = Half throttle setting (on LPG fuel only);
#   1.7) 7 = Full throttle setting (on LPG fuel only).
# 2) Fuel: The type of fuel that produced the flame, which may be: Gasoline, Kerosene, Thinner or LPG.
# 3) Distance: Distance of flame to collimator output, in centimeters.
# 4) Decibel: Sound pressure level, in decibels.
# 5) Airflow: Airflow created by the sound wave, in meters per second.
# 6) Frequency: Low frequency range, in hertz.
