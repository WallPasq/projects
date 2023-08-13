# Project developed during the course Big Data Analytics with R and Microsoft Azure Machine Learning,
# from the Data Science Academy: https://www.datascienceacademy.com.br/course/analise-de-dados-com-r

# Import the necessary libraries, the dataset and realize data munging

library(tidyr)
library(dplyr)
library(zoo)
library(purrr)
library(ggplot2)
library(countrycode)

# Data source: https://data.world/laurel/world-happiness-report-data

db <- read.csv('dataset.csv', stringsAsFactors=TRUE)
db$year = factor(db$year, ordered=TRUE)
View(db)
str(db)
summary(db)
sum(is.na(db))

# There are missing values in the dataset, in all columns except Country.name, year and Life.Ladder.
# Since we have a time series, to solve this, I built a function that parses the following:
# 1) If all column observations for the country are missing, then the value remains NA;
# 2) Otherwise, if there are non-null observations before and after the null observation
#    (in order of year), then the null value will be replaced by the average between the
#    first previous and subsequent non-null values;
# 3) If there are only null observations before the observation, then the null value
#    will be replaced by the first subsequent non-null observation;
# 4) If there are only null observations after the observation, then the null value
#    will be replaced by the first previous non-null observation.

locf_adj <- function(column) {
  locf_lag <- tryCatch(na.locf(column), error=function(e) return(NA))
  locf_lead <- tryCatch(na.locf(column, fromLast=TRUE), error=function(e) return(NA))
  locf_lag <- ifelse(is.na(locf_lag), locf_lead, locf_lag)
  locf_lead <- ifelse(is.na(locf_lead), locf_lag, locf_lead)
  return(mean(c(locf_lag, locf_lead)))
}

db_v2 <- db %>%
  arrange(Country.name, year) %>%
  group_by(Country.name) %>%
  mutate(across(names(db)[4:length(db)], ~ifelse(is.na(.), locf_adj(.), .))) %>%
  ungroup()

# Ensuring difference between missing values from db and db_v2

# Column Log.GDP.per.capita

db %>% 
  arrange(Country.name, year) %>%
  group_by(Country.name) %>% 
  filter(any(is.na(Log.GDP.per.capita))) %>% 
  select(Country.name, year, Log.GDP.per.capita) %>% 
  ungroup() %>% 
  View()

db_v2 %>% 
  arrange(Country.name, year) %>%
  group_by(Country.name) %>% 
  filter(any(is.na(Log.GDP.per.capita))) %>% 
  select(Country.name, year, Log.GDP.per.capita) %>% 
  ungroup() %>% 
  View()

# Column Social.support

db %>% 
  arrange(Country.name, year) %>%
  group_by(Country.name) %>% 
  filter(any(is.na(Social.support))) %>% 
  select(Country.name, year, Social.support) %>% 
  ungroup() %>% 
  View()

db_v2 %>% 
  arrange(Country.name, year) %>%
  group_by(Country.name) %>% 
  filter(any(is.na(Social.support))) %>% 
  select(Country.name, year, Social.support) %>% 
  ungroup() %>% 
  View()

# All columns
summary(db)
summary(db_v2)

# The remaining missing values are only when all country observations are null
# for that variable. To solve this, let's replace that with the yearly mean of
# the country's region variable.

db_v2$region <- countrycode(sourcevar=as.character(db_v2$Country.name),
                            origin='country.name',
                            destination='region')
db_v2$region <- factor(db_v2$region)

db_v2 <- 
  db_v2 %>% 
  group_by(year, region) %>% 
  mutate(across(names(db_v2)[4:(length(db_v2)-1)], ~ifelse(is.na(.), mean(., na.rm=TRUE), .))) %>% 
  ungroup()

summary(db_v2)
sum(is.na(db_v2))

rm(db)

# The problem with missing values is solved! Let's analyse the data.
# I will answer five questions:

# 1) The increase in a country's GDP per capita positively affects life expectancy
#    of citizens at birth? What is the correlation between these two variables?

cor_label <- function(x) {
  ifelse(is.na(x), '0 - not enough data to analyze correlation',
  ifelse(x <= -0.9, '1 - strong negative correlation',
  ifelse(x <= -0.5, '2 - slight negative correlation',
  ifelse(x <= -0.2, '3 - very weak negative correlation',
  ifelse(x < 0.2, '4 - no correlation',
  ifelse(x < 0.5, '5 - very weak positive correlation',
  ifelse(x < 0.9, '6 - slight positive correlation',
                  '7 - strong positive correlation')))))))
}

cor_analysis <- function(df, var1, var2, group=NULL) {
  
  results <- vector('list', length=3)
  names(results) <- c('scatter_plot', 'group_analysis', 'correlation')
  
  results$scatter_plot <- ggplot(df, aes(x={{var1}}, y={{var2}})) +
    geom_point() + geom_smooth(method=lm)
  
  if (!missing(group)) {
    results$group_analysis <-  df %>%
      group_by({{group}}) %>% 
      summarise(correlation = cor({{var1}}, {{var2}})) %>% 
      ungroup() %>% 
      mutate(
        condition = cor_label(correlation),
        general_amount = n()) %>% 
      group_by(condition) %>% 
      summarise(amount = n(),
                perc = n() / mean(general_amount)) %>% 
      ungroup() %>% 
      arrange(condition)
  } else {
    results$group_analysis <- 'No group passed as parameter'
  }
  
  results$correlation <- df %>% 
    summarise(correlation = cor({{var1}}, {{var2}})) %>% 
    pull(correlation)
  
  return(results)
}

cor_analysis(db_v2, Log.GDP.per.capita, Healthy.life.expectancy.at.birth, Country.name)

# Yes, the most part of countries (122 countries - 73.5%) have a strong (>= 90%)
# or slight (> 50% and < 90%) positive correlation between GDP per capita and
# expectancy of citizens at birth, Of which 74 countries (60.7%) have strong
# positive correlation. Overall, the correlation is equal to 84.56%.

# 2) There is a correlation between life scale and general public awareness about
#    corruption in business and government? What is the correlation between these
#    two variables?

cor_analysis(db_v2, Life.Ladder, Perceptions.of.corruption, Country.name)

# It's not possible to say that with these data. Although, in general, it presents
# a very weak negative correlation (-44%), the most part of countries
# (105 countries - 63.3%) are between a very weak negative correlation and a very
# weak positive correlation (> -50% and < 50%). Just 42 countries (25.3%) have
# a strong or slight negative correlation.

# 3) The increase in life scale has some effect on average happiness among the
#    public in general? What is the correlation between these two variables?

cor_analysis(db_v2, Life.Ladder, Positive.affect, Country.name)

# There is a slight relationship between life scale and average happiness.
# In general, it presents a slight positive correlation (53%), and the most part
# of countries (86 countries - 51.8%) have a correlation greater than 20%.
# Of these, 52 countries (60.4%) have a slight positive correlation (< 90%), and
# only 3 countries have a correlation above 90%.

# 4) The country with the lowest social support index has the highest perception
#    of corruption in relation to companies and the government in the country?

db_v2 %>%
  group_by(Country.name) %>% 
  summarise(
    Social.support = median(Social.support),
    Perceptions.of.corruption = median(Perceptions.of.corruption)) %>% 
  mutate(
    Rank.Social.support = dense_rank(Social.support),
    Rank.Perceptions.of.corruption = dense_rank(desc(Perceptions.of.corruption))) %>% 
  arrange(Rank.Social.support, Rank.Perceptions.of.corruption) %>% 
  View()

# No. In a ranking of the country with the lowest social support index to the
# highest (with 166 countries), the Central African Republic has the lowest social
# support, and it is the 42nd with the highest perception of corruption. The country
# with the highest perception of corruption is Romania, which is the 57th country
# with the lowest social support.

# 5) Are generous people happier?

cor_analysis(db_v2, Generosity, Positive.affect, Country.name)

# It's not possible to say that with these data. Although, in general, it presents
# a very weak positive correlation (35.3%), the most part of countries
# (98 countries - 59.1%) are between a very weak negative correlation and a very
# weak positive correlation (> -50% and < 50%). Just 50 countries (25.5%) have a
# strong or slight positive correlation.
