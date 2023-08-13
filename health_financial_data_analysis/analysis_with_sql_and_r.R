# Import the necessary libraries, the dataset and realize data munging

library(sqldf)
library(dplyr)
library(ggplot2)

##### About data #####

# Data sources: 
# https://healthdata.gov/
# https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Provider-Cost-Report/HospitalCostPUF

# Data dictionary:
# AGE           | Patient's age.
# FEMALE        | Binary variable that indicates whether the patient is female.
# LOS           | Length of stay of the patient.
# RACE          | Patient's race.
# TOTCHG        | Cost of hospitalization.
# APRDRG        | Refined patient diagnostic group.

db <- read.csv('dataset.csv')
View(db)
str(db)
summary(db)
sum(is.na(db))
colSums(is.na(db))

names(db) <- unlist(lapply(names(db), tolower))

# There is only a observation without race information. Let's delete it.

db <- na.omit(db)

##### Exploratory data analysis #####

# I will answer ten questions about the data using SQL language.

# 1) How many races are represented in the dataset?

sqldf("SELECT COUNT(DISTINCT race) total_races FROM db")

# There are six races.

# 2) What's the average age of patients?

sqldf("SELECT AVG(age) age_mean FROM db")

# The average age is five years.

# 3) What's the age mode of the patients?

sqldf("SELECT age, COUNT(age) total FROM db
       GROUP BY 1 ORDER BY 2 DESC LIMIT 1")

# The age mode is zero years, with 306 cases.

# 4) What's the variance of the age column?

sqldf("SELECT VARIANCE(age) age_variance FROM db")

# The variance is 48.34.

# 5) What's the total expenditure on hospital admissions by age?

sqldf("SELECT age, SUM(totchg) total_expenditure FROM db
       GROUP BY 1 ORDER BY 2 DESC")

# Total expenditures by age, ordered from highest expenditure to lowest expenditure.
# age   total_expenditure
# 0     676,962
# 17    174,777
# 15    111,747
# 16    69,149
# 14    64,643
# 12    54,912
# 1     37,744
# 13    31,135
# 3     30,550
# 10    24,469
# 9     21,147
# 5     18,507
# 6     17,928
# 4     15,992
# 11    14,250
# 7     10,087
# 2     7,298
# 8     4,741

# 6) What age generates the highest total expenditure on hospital admissions?

sqldf("SELECT age, SUM(totchg) total_expenditure FROM db
       GROUP BY 1 ORDER BY 2 DESC LIMIT 1")

# Zero years, with a total expenditure of $676,962.

# 7) What's the total expenditure on hospital admissions by gender?

sqldf("SELECT
         CASE
           WHEN female = 0 THEN 'male'
           ELSE 'female'
         END gender
        ,SUM(totchg) total_expenditure FROM db
       GROUP BY 1 ORDER BY 2 DESC")

# Total expenditure by gender
# gender    total_expenditure
# male      735,391
# female    650,647

# 8) What's the average expenditure on hospital admissions by patient race?

sqldf("SELECT race, AVG(totchg) average_expenditure FROM db
       GROUP BY 1 ORDER BY 2 DESC")

# Average expenditure by race, ordered from highest expenditure to lowest expenditure.
# race  average_expenditure
# 2     4,202.17
# 3     3,041.00
# 1     2,772.67
# 4     2,344.67
# 5     2,026.67
# 6     1,349.00

# 9) For patients over 10 years old, what is the average total expenditure on
#    hospital admissions?

sqldf("SELECT AVG(totchg) mean_expenditure FROM db
       WHERE age > 10")

# The average expenditure for patients over 10 years old is $3,213.66.

# 10) Considering the previous question, which age has an average expenditure
#     greater than 3,000?

sqldf("SELECT age, AVG(totchg) mean_expenditure FROM db
       WHERE age > 10
       GROUP BY 1
       HAVING mean_expenditure > 3000")

# Ages 12, 15 and 17 has an average expenditure greater than $3,000.

##### Analyzes using statistics (with linear regression) #####

# I will answer seven questions about the data using statistics.

# 11) What's the age distribution of patients attending the hospital?

bins <- 10
min_x <- min(db$age)
max_x <- max(db$age)

breaks <- hist(db$age, breaks=bins, plot=FALSE)
counts <- breaks$counts
mids <- breaks$mids
breaks <- breaks$breaks
breaks[1] <- min_x
breaks[length(breaks)] <- max_x

hist(db$age, breaks=bins, xlab='Age', border='black', xaxt='n', xlim=c(min_x, max_x),
     main='Histogram of the age distribution of patients who attended the hospital')
axis(1, at=breaks)
text(mids, counts + 7.5, col='black',
     paste(counts,' (', round(100 * counts/sum(counts), 0), '%)', sep=''))

# Children between 0 and 2 years old represent the age group that mostly attends
# the hospital (64% of cases).

# 12) Which age group has the highest total expenditure in the hospital?

db %>%
  mutate(age_cut=cut(age, breaks, include.lowest=TRUE)) %>% 
  aggregate(totchg ~ age_cut, FUN=sum) %>%
  arrange(desc(totchg)) %>% 
  head(1)

# Children between 0 and 2 years has the highest expenditure ($722,004).

# 13) Which group based on diagnosis has the highest total expenditure in the
#     hospital?

db %>%
  aggregate(totchg ~ aprdrg, FUN=sum) %>%
  arrange(desc(totchg)) %>% 
  head(1)

# The group 640 has the highest total expenditure ($436,822).

# 14) Is the patient's race related to the total expenditure spend on hospital
#     admissions?

# We will use the ANOVA test to verify this.
# H0: Race does not influence total expenditure.
# Ha: Race influence total expenditure.

summary(aov(totchg ~ race, data=db))

# Since the p-value is greater than 0.05, we fail to reject the null hypothesis
# and therefore assume that race does not influence total expenditure.

# 15) The combination of age and gender of patients influences the total
#     expenditure on hospital admissions?

# We will use the ANOVA test to verify this.
# H0: Age and gender does not influence total expenditure.
# Ha: Age and gender influence total expenditure.

summary(aov(totchg ~ age + female, data=db))

# As the p-value is less than 0.05 in both cases, we reject the null hypothesis
# and therefore assume that age and gender influence total expenditure.

# 16) As the length of stay is a crucial factor for hospitalized patients,
#     we wish to find out whether length of stay cand be predicted from age,
#     gender and race?

# We will use a linear regression model to answer this question.
# H0: Age, gender and race do not, by themselves, predict total expenditure.
# Ha: Age, gender and race predict total expenditure.

lm_model_v1 <- lm(los ~ age + female + race, data=db)
summary(lm_model_v1)

# In both cases, p-value is greater than 0.05, so we fail to reject the null
# hyphotesis. Therefore, we assume that age, gender and race do not, by
# themselses, predict length of stay.

# 17) Which variables have the greatest impact on hospitalization expenditure?

# We will use a liner regression model to answer this question.

lm_model_v2 <- lm(totchg ~ ., data=db)
summary(lm_model_v2)

# Gender and race have p-value greater than 0.05. Let's try to remove these
# variables from the model.

lm_model_v3 <- lm(totchg ~ age + los + aprdrg, data=db)
summary(lm_model_v3)

# Both variables have p-value less than 0.05, but this model does not have R²
# greater than the last model. As the diagnostic group has a negative t-value,
# let's try to remove it from the model.

lm_model_v4 <- lm(totchg ~ age + los, data=db)
summary(lm_model_v4)

# The R² decreased. Version 3 of the model was the best. Therefore, we consider
# that age, length of stay and diagnosis group are the variables that best explain
# hospitalization expenditure.
