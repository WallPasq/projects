# Project developed during the course Big Data Analytics with R and Microsoft Azure Machine Learning,
# from the Data Science Academy: https://www.datascienceacademy.com.br/course/analise-de-dados-com-r

# Import the necessary libraries

library(h2o)
library(tidyverse)
library(ggbeeswarm)

##### About data #####

# Let's assume that a company produces hospital materials through one of its factories
# in Brazil. Every factory has a variety of industrial equipment that periodically
# requires maintenance. The idea is to have a Machine Learning model able to predict
# when each machine will need maintenance and thus avoid downtime unscheduled.
# But before using a predictive model, top management needs to understand how the model
# makes the predictions and which metrics have the greatest impact on the model's prediction.

# To make this work, let's first create a mass of dummy data, which represents real data.
# Note: The intent here is to show skills in using AutoML to create ML models and
# SHAP values for model explainability. So, in order to have a controlled environment,
# I'm creating a mass of dummy data. If you want to see projects where I work with real
# data, some of them are at: https://github.com/WallPasq/projects

# Data dictionary

# oef         | Overall Equipment Effectiveness: This is a measure of productivity,
#             | which describes the part the time a machine works at peak performance.
#             | The metric is a product of machine availability, performance and quality.
# ------------|--------------------------------------------------------------------------
# fpy         | First Pass Yield: It is the portion of the product that leaves the
#             | production line, and that are free of defects and meet specifications
#             | without the need for any rectification work.
# ------------|--------------------------------------------------------------------------
# ecu         | Energy cost per unit: This is the cost of electricity, steam, oil
#             | or gas needed to produce a certain unit of product in the factory.
# ------------|--------------------------------------------------------------------------
# epr         | Equipment priority when entering the maintenance period (Low, Medium,
#             | High).
# ------------|--------------------------------------------------------------------------
# apm         | The amount of product a machine produces during a specific period.
#             | This metric can also be applied to the entire production line to
#             | check its efficiency.
# ------------|--------------------------------------------------------------------------
# main        | 0 means that the equipment does not require maintenance (no);
#             | 1 means that the equipment requires maintenance (yes).

##### Creating a mass of dummy data #####

set.seed(42)
db <- tibble(oef = c(rnorm(1000), rnorm(1000, 0.25)),
             fpy = runif(2000),
             ecu = rf(2000, df1=5, df2=2),
             epr = factor(c(sample(rep(c('Low', 'Medium', 'High'), c(300, 300, 400))),
                            sample(c('Low', 'Medium', 'High'), 1000, prob=c(0.25, 0.25, 0.5),
                                   replace=TRUE))),
             apm = rnorm(2000),
             main = factor(rep(c(0, 1), c(1050, 950))))
View(db)
str(db)

# Checks the distribution of the target variable in the dataset.

db %>% 
  group_by(main) %>% 
  summarise(freq = n(),
            perc = n() / nrow(db))

# Checks the distribution of the epr variable.

db %>% 
  group_by(epr) %>% 
  summarise(freq = n(),
            perc = n() / nrow(db))

##### Using AutoML to create multiple ML models #####

# Let's use h2o framework to apply AutoML.
# h2o needs to be started, and it uses the JVM (Java Virtual Machine) to run the commands.
# It also requires the data to be in h2o dataframe format.

h2o.init()
h2o_frame <- as.h2o(db)
class(h2o_frame)
head(h2o_frame)
str(h2o_frame)

# Splitting data into train and test
h2o_frame_split <- h2o.splitFrame(h2o_frame, ratios=0.77)
names(h2o_frame_split) <- c('train', 'test')
head(h2o_frame_split)

# AutoML model
automl_model <- h2o.automl(y='main',
                           balance_classes=TRUE,
                           training_frame=h2o_frame_split$train,
                           nfolds=4,
                           leaderboard_frame=h2o_frame_split$test,
                           max_runtime_secs=60 * 2,
                           include_algos=c('XGBoost', 'GBM'),
                           sort_metric='AUC')

# Extracts the model leaderboard (list of trained models)
leaderboard_automl <- as.data.frame(automl_model@leaderboard)
View(leaderboard_automl)

# Extracts the leader (model with better performance)
leader_automl <- automl_model@leader
View(leader_automl)

##### Answering business question #####

# The top management question is: What factors (metrics) most contribute to explain the
# behavior of the need for maintenance in a piece of equipment? Why?

# Let's answer that, using SHAP values.
# First, let's extract the contribution of each variable to the predictions of the best model.
# Note: Extracted values are SHAP values.

var_contrib <- predict_contributions.H2OModel(leader_automl, h2o_frame_split$test)

# Creating a dataframe with organized data for plotting.

db_var_contrib <- var_contrib %>% 
  as.data.frame() %>% 
  select(-BiasTerm) %>% 
  gather(feature, shap_value) %>% 
  group_by(feature) %>% 
  mutate(shap_importance=mean(abs(shap_value)),
         shap_force=mean(shap_value)) %>% 
  ungroup()

# Plotting the importance of each variable to predict the target variable.

db_var_contrib %>% 
  select(feature, shap_importance) %>% 
  distinct() %>% 
  ggplot(aes(x=reorder(feature, shap_importance), y=shap_importance)) +
    geom_col(fill='blue') +
    coord_flip() +
    xlab(NULL) +
    ylab('Mean value of SHAP metrics') +
    theme_minimal(15)

# Plotting the contribution of each variable to explain the target variable.
# I'm using ggbeeswarm::geom_quasirandom to offset points within categories to
# reduce overplotting.

ggplot(db_var_contrib, aes(x=shap_value, y=reorder(feature, shap_importance))) +
  ggbeeswarm::geom_quasirandom(groupOnX=FALSE, varwidth=TRUE, size=0.9, alpha=0.5, width=0.15) +
  xlab('Variable contribution') +
  ylab(NULL) +
  theme_minimal(15)

# The most important variables to predict if equipment needs maintenance is:
## epr: Equipment priority when entering the maintenance period;
## oef: Overall Equipment Effectiveness;
## fpy: First Pass Yield.

# If the equipment has a Low or Medium epr, the less likely it is to need maintenance;
# The greater the oef, the more likely it is to need maintenance;
# The greater the fpy, the lower the need for the equipment to need maintenance.

# The ecu (Energy cost per unit) and apm (The amount of product a machine produces
# during a specific period) have almost no relevance for explaining the need for
# equipment maintenance.
