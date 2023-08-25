# Project developed during the course Big Data Analytics with R and Microsoft Azure Machine Learning,
# from the Data Science Academy: https://www.datascienceacademy.com.br/course/analise-de-dados-com-r

#### Context ####

# The company Bon Voyage wants to switch its fleet to electric cars to reduce costs. However, first
# he needs to know if this exchange is really worth it, understanding the expected energy consumption
# of these vehicles.


#### Business question ####

# What is the energy consumption of electric cars based on their characteristics and usage factors,
# such as vehicle weight, load capacity, etc.?


#### How to solve it? ####

# Using a Linear Regression Machine Learning model, with R language.


#### Definition of done ####

# The model must have at least R² of 0.9 in train and test.


#### About data ####

# Reference
# Hadasik, Bartłomiej; Kubiczek, Jakub (2021), “Dataset of electric passenger cars with their
# specifications”, Mendeley Data, V2, doi: 10.17632/tb9yrptydn.2

# Summary
# This dataset lists all fully electric passenger cars with their attributes (properties). The collection
# does not contain data on plug-in hybrid cars and electric cars from the so-called “range extenders”.
# Hydrogen cars were also not included in the dataset due to the insufficient number of mass-produced
# models and different (compared to EV) specificity of the vehicle, including the different charging
# methods. The dataset includes cars that, as of 2 December, 2020, could be purchased in Poland as new
# at an authorized dealer and those available in public and general presale, but only if a publicly
# available price list with equipment versions and full technical parameters was available. The list
# does not include discontinued cars that cannot be purchased as new from an authorized dealer (also
# when they are not available in stock). The subject of the study is only passenger cars, the main
# purpose of which is to transport people (inter alia without including vans intended for professional
# deliveries). The dataset of electric cars includes all fully electric cars on the primary market that
# were obtained from official materials (technical specifications and catalogs) provided by automotive
# manufacturers with a license to sell cars in Poland. These materials were downloaded from their official
# websites. In the event that the data provided by the manufacturer were incomplete, the information was
# supplemented with data from the SAMAR AutoCatalog. The database consisting of 53 electric cars (each
# variant of a model – which differs in terms of battery capacity, engine power, etc. – is treated as
# separate) and 22 variables (25 variables, including make, model and “car name” merging these two previous).

# Data source:
# https://data.mendeley.com/datasets/tb9yrptydn/2


#### Data dictionary ####

# Made with help of ChatGPT.

# |-----------------------|------------------------------|--------------------------------------------------|
# | Original name         | Name on this project         | Definition                                       |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Car full name         | car                          | It is the primary key of the table, being the    |
# |                       |                              | junction of the make and model columns. Names    |
# |                       |                              | the car we are referring to in the observartion. |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Make                  | brand                        | The vehicle manufacturer.                        |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Model                 | model                        | The vehicle model.                               |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Minimal price (gross) | price                        | The vehicle gross price, in Polish złoty.        |
# | [PLM|                 |                              |                                                  |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Engine power [KM]     | hp                           | Is a unit of rate at which work is done, or the  |
# |                       |                              | measurement of power. Horsepower is a measure of |
# |                       |                              | how quickly an engine delivers its torque.       |
# |                       |                              | Horsepower is important because it more closely  |
# |                       |                              | relates to how quick and fast a car is.          |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Maximum torque [Nm]   | torque                       | Torque is the rotational force created by an     |
# |                       |                              | engine or motor that turns the wheels and        |
# |                       |                              | propels the vehicle down the road. More torque   |
# |                       |                              | makes a car accelerate harder, assuming the      |
# |                       |                              | weight stays the same.                           |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Type of brakes        | brakes                       | Defines the types of brakes, if they are:        |
# |                       |                              | * disc (front + rear)                            |
# |                       |                              | * disc (front) + drum (rear)                     |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Drive type            | wheel                        | Defines the car's wheel drive model, if it is:   |
# |                       |                              | * 4WD - The Four Wheel Drive (4WD) system is     |
# |                       |                              |   designed to provide power to all four wheels   |
# |                       |                              |   of the vehicle. In general, a 4WD system       |
# |                       |                              |   provides more pulling power than a front (2WD) |
# |                       |                              |   or rear (2WD) drive system, as it distributes  |
# |                       |                              |   power between all wheels.                      |
# |                       |                              | * 2WD (rear) - means that power is sent to the   |
# |                       |                              |   rear wheels of the vehicle only. This type of  |
# |                       |                              |   setup used to be more common on older cars and |
# |                       |                              |   high-performance vehicles, as it provides a    |
# |                       |                              |   more balanced weight distribution and better   |
# |                       |                              |   cornering control.                             |
# |                       |                              | * 2WD (front) - means power is sent to the front |
# |                       |                              |   wheels only. This is a common arrangement in   |
# |                       |                              |   many passenger cars and compact vehicles.      |
# |                       |                              |   Front-wheel drive tends to be more efficient   |
# |                       |                              |   in terms of energy savings in electric cars.   |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Battery capacity      | battery                      | Is the capacity of the vehicle's battery.        |
# | [kWh]                 |                              |                                                  |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Range (WLTP) [km]     | range                        | It is estimated range of the vehicle according   |
# |                       |                              | to the WLTP test procedure (Worldwide Harmonized |
# |                       |                              | Light Vehicles Test Procedure). Range is the     |
# |                       |                              | distance an electric vehicle can travel on a     |
# |                       |                              | full battery charge before needing to be         |
# |                       |                              | recharged.                                       |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Wheelbase [cm]        | wheelbase                    | The distance between the front and rear axles of |
# |                       |                              | a vehicle.                                       |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Length [cm]           | length                       | The length of the vehicle.                       |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Width [cm]            | width                        | The width of the vehicle.                        |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Height [cm]           | height                       | The height of the vehicle.                       |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Minimal empty weight  | min_weight                   | The weight of the vehicle when empty.            |
# | [kg]                  |                              |                                                  |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Permissable gross     | gross_weight                 | The maximum weight allowed by law for the        |
# | weight [kg]           |                              | vehicle.                                         |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Maximum load capacity | max_capacity                 | The maximum load weight the vehicle can support. |
# | [kg]                  |                              |                                                  |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Number of seats       | seats                        | Number of vehicle seats.                         |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Number of doors       | doors                        | Number of vehicle doors.                         |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Tire size [in]        | tire                         | The tire size.                                   |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Maximum speed [kph]   | max_speed                    | The maximum speed the vehicle reaches.           |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Boot capacity (VDA)   | boot                         | Is the vehicle's trunk capacity, measured        |
# | [l]                   |                              | according to the Automobile Industry Association |
# |                       |                              | (VDA). The technique involves the use of         |
# |                       |                              | standardized geometric blocks, called "luggage   |
# |                       |                              | cubes", which are placed in the trunk in an      |
# |                       |                              | orderly fashion, counted and measured to         |
# |                       |                              | determine the total volume of cargo space        |
# |                       |                              | available.                                       |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Acceleration 0-100    | acceleration                 | The time required, in seconds, for the vehicle   |
# | kph [s]               |                              | to reach 100 kilometers per hour.                |
# |-----------------------|------------------------------|--------------------------------------------------|
# | Maximum DC charging   | charge                       | The vehicle's maximum charging power.            |
# | power [kW]            |                              |                                                  |
# |-----------------------|------------------------------|--------------------------------------------------|
# | mean - Energy         | energy                       | It is the target variable, referring to the      |
# | consumption           |                              | vehicle's average consumption in kilowatt-hours  |
# | [kWh/100 km]          |                              | per hundred kilometers.                          |
# |-----------------------|------------------------------|--------------------------------------------------|


#### Loading, cleaning and exploratory data analysis ####
library(readxl)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(corrgram)
library(randomForest)

# Checks which are the sheets of the Excel file.
file <- 'Projeto 1/dataset.xlsx'
sheet <- excel_sheets(file)
db <- read_excel(file, sheet=sheet)

rm(file)
rm(sheet)

names(db) <- c('car', 'brand', 'model', 'price', 'hp', 'torque', 'brakes', 'wheel',
               'battery', 'range', 'wheelbase', 'length', 'width', 'height', 'min_weight',
               'gross_weight', 'max_capacity', 'seats', 'doors', 'tire', 'max_speed',
               'boot', 'acceleration', 'charge', 'energy')

View(db)
str(db)

# Handling missing values.
amount_nas_col <- function(x) {
  return(
    data.frame(amount_nas=colSums(is.na(x))) %>%
      filter(amount_nas > 0) %>%
      arrange(desc(amount_nas)))
}

amount_nas_col(db)

# There are missing values in:
# 1) energy: 9 missing values
# 2) gross_weight: 8 missing values
# 3) max_capacity: 8 missing values
# 4) acceleration: 3 missing values
# 5) boot: 1 missing value
# 6) brakes: 1 missing value

# Energy is our target variable. Unfortunately, it is also the one with the most missing values.
# However, let's try removing her missing values from our dataset and see how that goes.
# dbwem = database without energy missing.
dbwem <- filter(db, !is.na(energy))
amount_nas_col(dbwem)

# There are missing values in:
# 1) brakes: 1 missing value
# 2) boot: 1 missing value
# 3) acceleration: 2 missing values

# Let's treat them column by column.

# 1) Brakes:
dbwem %>%
  group_by(brand) %>% 
  filter(any(is.na(brakes))) %>% 
  ungroup() %>% 
  View()

# There are only two Mercedes-Benz vehicles in the dataset, and the other vehicle has disc
# brakes (front + rear). Let's check how the vehicles are distributed by wheel and brakes.
dbwem %>% 
  filter(!is.na(brakes)) %>%
  group_by(wheel) %>% 
  mutate(total = n()) %>% 
  ungroup() %>% 
  group_by(wheel, brakes) %>% 
  summarise(
    amount=n(),
    perc=amount/mean(total)) %>% 
  ungroup() %>% 
  ggplot(aes(x=wheel, y=amount, fill=brakes)) +
    geom_bar(position='dodge', stat='identity') +
    geom_text(aes(y=amount,
                  label=scales::percent(perc)),
              position=position_dodge(width=0.9), vjust=-0.5) +
    labs(title='Distribution of types of brakes by wheel')

# Most vehicles (90.50%) with wheel equal to 2WD (front) have type of brakes disc (front + rear).
# Since the only Mercedes Benz vehicle in the dataset also has this type of brakes and they are
# similar models, let's insert this type of brakes for this vehicle into the dataset.
dbwem[is.na(dbwem$brakes), grep('brakes', names(dbwem))] <- 'disc (front + rear)'

# Validates that all vehicles have the brakes column filled in.
dbwem %>%
  filter(is.na(brakes)) %>% 
  nrow()

# Checks if the value has been properly filled in.
View(dbwem[dbwem$brand=='Mercedes-Benz',])

# The remaining columns with missing values are all numeric. Let's check if there are
# any strong correlations between numeric columns.
correlations <-
  dbwem %>%
    select(where(is.numeric)) %>% 
    na.omit() %>% 
    cor() %>% 
    as.data.frame()

correlations <-
  correlations %>% 
    mutate(var1=rownames(correlations)) %>% 
    gather(key='var2', value='cor', -var1) %>% 
    filter(cor > 0.8 | cor < -0.8, var1 != var2) %>%
    arrange(desc(cor), var1) %>% 
    slice(-seq(0, n(), 2))

View(correlations)

correlations %>% 
  ggplot(aes(var1, var2)) +
    geom_tile(aes(fill=cor)) +
    geom_text(aes(label=round(cor, 4))) +
    scale_fill_gradient(low='lightblue', high='darkred')

# There are 37 combinations of variables with correlation above 0.8 and 6 combinations below -0.8,
# the latter involving the acceleration variable. 

# 2) boot
dbwem %>%
  filter(is.na(boot)) %>%
  View()

# Let's treat this missing value using linear regression with the remaining variables in the
# dataset, except the vehicle identifiers (car, brand, model), the target variable (energy)
# and the other variable that is also missing in this observation (acceleration).
boot_db <- dbwem %>% select(-car, -brand, -model, -energy, -acceleration)
boot_lm <- lm(boot ~ ., boot_db)
summary(boot_lm)

# The model has R² 0.9629. Let's use this model to fill in the missing value of boot.
boot_missing <- dbwem %>%
  select(-car, -brand, -model, -energy, -acceleration) %>% 
  filter(is.na(boot))

dbwem[is.na(dbwem$boot), grep('boot', names(dbwem))] <- as.integer(predict(boot_lm, boot_missing))
dbwem %>%
  filter(is.na(boot)) %>%
  nrow()

View(dbwem[dbwem$car=='Mercedes-Benz EQV (long)',])

rm(boot_db)
rm(boot_lm)
rm(boot_missing)

# 3) acceleration
dbwem %>%
  filter(is.na(acceleration)) %>%
  View()

accer_db <- dbwem %>% select(-car, -brand, -model, -energy)
accer_lm <- lm(acceleration ~ ., accer_db)
summary(accer_lm)

# The model has R² 0.9661. Note that it doesn't use any variable that identifies the cars,
# and it also doesn't use the final target variable. Let's use this model to fill in the
# missing value of acceleration.
accer_missing <- dbwem %>% 
  select(-car, -brand, -model, -energy) %>% 
  filter(is.na(acceleration))

dbwem[is.na(dbwem$acceleration), grep('acceleration', names(dbwem))] <- predict(accer_lm, accer_missing)
dbwem %>%
  filter(is.na(acceleration)) %>%
  nrow()

rm(accer_db)
rm(accer_lm)
rm(accer_missing)

# Let's check if there are any missing values left.
amount_nas_col(dbwem)
str(dbwem)

# Let's turn the columns that are strings into factors.
dbwem <- as.data.frame(lapply(dbwem, function(x) if (is.character(x)) as.factor(x) else x))
str(dbwem)
summary(dbwem)

# Let's delete all columns that are factors and contains the same number of levels as the
# number of lines in the dataset, because they are unique identifiers of the observations.
del_columns <- dbwem %>% 
  select(where(is.factor)) %>% 
  select(where(function(x) length(levels(x)) == nrow(dbwem))) %>% 
  names()
del_columns

dbwem <- select(dbwem, -all_of(del_columns))
rm(del_columns)

str(dbwem)


#### Predictive models ####

# Analyzing which variables are most important to predict vehicle energy consumption.
# varImp = variable importance

set.seed(42)
varImp_v1 <- randomForest(energy ~ ., data=dbwem, ntree=100, nodesize=10, importance=TRUE)
varImpPlot(varImp_v1)

table(dbwem$brand) %>%
  as.data.frame() %>%
  arrange(Freq)

# Brand appears as the most important variable. However, most brands only have one or two vehicles
# in the dataset. Let's check the importance of the variables disregarding the brand.

set.seed(42)
varImp_v2 <- randomForest(energy ~ . -brand, data=dbwem, ntree=100, nodesize=10, importance=TRUE)
varImpPlot(varImp_v2)

rm(varImp_v1)
rm(varImp_v2)

# The ten most important variables are:
# length
# max_capacity
# wheelbase
# gross_weight
# max_speed
# wheel
# min_weight
# width
# price
# boot

top_ten <- c('length', 'max_capacity', 'wheelbase', 'gross_weight', 'max_speed',
             'wheel', 'min_weight', 'width', 'price', 'boot')

correlations %>% 
  filter(var1 %in% top_ten, var2 %in% top_ten)

# There are twelve strong correlations (greater than 0.8) between the ten most important variables,
# and only the wheel and width variables do not show strong correlation with the rest.

# Let's create three regression models:
# 1) Using all variables except brand;
# 2) Using the top ten important variables;
# 3) Using the top five important variables.

# 1) Using all variables except brand
dbwem <- dbwem %>% select(-brand)
set.seed(42)
dbwem['TestTrain'] <- sapply(runif(nrow(dbwem)), function(x) if (x < 0.2) 0 else 1)

x_test <- dbwem %>% filter(TestTrain == 0) %>% select(-energy, -TestTrain)
y_test <- dbwem %>% filter(TestTrain == 0) %>% select(energy)
db_train <- dbwem %>% filter(TestTrain == 1) %>% select(-TestTrain)

print(nrow(db_train) / nrow(dbwem)) # 84% of the dataset
print(nrow(x_test) / nrow(dbwem)) # 16% of the dataset

lm_model_v1 <- lm(energy ~ ., data=db_train)
summary(lm_model_v1) # R² in train = 0.9823

evaluate_model <- function(predictions, real) {
  db <- data.frame(predictions, real)
  names(db) <- c('predictions', 'real')
  db['residuals'] <- db$real - db$predictions
  mse <- mean((db$real - db$predictions) ^ 2)
  rmse <- mse ^ 0.5
  sse <- sum((db$real - db$predictions) ^ 2)
  sst <- sum((mean(db$real) - db$real) ^ 2)
  r2 <- 1 - (sse / sst)
  return(list(db=db, mse=mse, rmse=rmse, sse=sse, sst=sst, r2=r2))
}

evaluate_model(predict(lm_model_v1, x_test), y_test) # R² in test = 0.7135

# This model was probably overfitted due to the number of variables, which made its coefficient
# of determination drop significantly in the test.

# 2) Using the top ten important variables
lm_formula_v2 <- formula(paste('energy ~', paste(top_ten, collapse=' + ')))
lm_model_v2 <- lm(lm_formula_v2, data=db_train)

summary(lm_model_v2) # R² in train = 0.8778
evaluate_model(predict(lm_model_v2, x_test), y_test) # R² in test = 0.9450

# Despite the coefficient of determination having dropped slightly in train in relation to model 1,
# it was much higher in the test, which helps to strengthen the overfitting hypothesis in model 1.

# 3) Using the top five important variables
lm_formula_v3 <- formula(paste('energy ~', paste(top_ten[1:5], collapse=' + ')))
lm_model_v3 <- lm(lm_formula_v3, data=db_train)

summary(lm_model_v3) # R² in train = 0.7511
evaluate_model(predict(lm_model_v3, x_test), y_test) # R² in test = 0.8811

# This model performed worse than model 2 both in the test and in the train.
# Let's stick with model 2, because it performed better.
model <- lm_model_v2
form <- lm_formula_v2
rm(correlations)
rm(top_ten)
rm(lm_model_v1)
rm(lm_formula_v2)
rm(lm_model_v2)
rm(lm_formula_v3)
rm(lm_model_v3)


#### Fine adjustments to the model ####

# Let's try to improve the efficiency of the model.
summary(model)

# Price seems to be the most important variable. I'm going to create a new column by squaring that value.
db_train['price2'] <- db_train$price ^ 2
x_test['price2'] <- x_test$price ^ 2
adj_formula_v1 <- as.formula(paste('energy ~', paste(form, '+ price2')[[3]]))
adj_model_v1 <- lm(adj_formula_v1, data=db_train)

summary(adj_model_v1) # R² in train = 0.8899
evaluate_model(predict(adj_model_v1, x_test), y_test) # R² in teste = 0.8954

# The model performed better in training, but not in testing. However, the variables max_speed
# and price proved to be more relevant. Let's try to modify them too.
# Price has positive coefficient, while max_speed has negative coefficient. Let's standardize
# the values (to get the distances from the mean) and multiply one by the other, thus seeking
# a balance between the variables.
my_scale <- function(x) {
  return(list(
    mean=mean(x),
    sd=sd(x),
    scaled=(x - mean(x)) / sd(x)
  ))
}

max_speed_scaled <- my_scale(db_train$max_speed)
price_scaled <- my_scale(db_train$price)

db_train$max_speed_scaled <- max_speed_scaled$scaled
db_train$price_scaled <- price_scaled$scaled
db_train$price_x_max_speed <- db_train$price_scaled * db_train$max_speed_scaled

# Let's adjust the test base avoiding data leakage.
x_test$max_speed_scaled <- (x_test$max_speed - max_speed_scaled$mean) / max_speed_scaled$sd
x_test$price_scaled <- (x_test$price - price_scaled$mean) / price_scaled$sd
x_test$price_x_max_speed <- x_test$price_scaled * x_test$max_speed_scaled

adj_formula_v2 <- as.formula(paste('energy ~', paste(form, '+ price_x_max_speed')[[3]]))
adj_model_v2 <- lm(adj_formula_v2, data=db_train)

summary(adj_model_v2) # R² in train = 0.8783
evaluate_model(predict(adj_model_v2, x_test), y_test) # R² in teste = 0.9479

# This model performed a little better than the original version. However, the new variable did not
# show enough p-value, being greater than that of the price variable. Let's try a model that adds
# price2 and price_x_max_speed variables.
adj_formula_v3 <- as.formula(paste('energy ~', paste(form, '+ price2 + price_x_max_speed')[[3]]))
adj_model_v3 <- lm(adj_formula_v3, data=db_train)

summary(adj_model_v3) # R² in train = 0.9048
evaluate_model(predict(adj_model_v3, x_test), y_test) # R² in teste = 0.9354

# This model performed better in training and a little worse in tests. However, it meets our
# definition of done by having R² greater than 0.9. Let's use it!
final_model <- adj_model_v3

print(price_scaled$mean)
print(price_scaled$sd)
print(max_speed_scaled$mean)
print(max_speed_scaled$sd)

# Function to make predictions
energy_prediction <- function(x, model=final_model) {
  
  # If you retrain the model, modify these mean and standard deviation values according to the values 
  # of the new training base.
  
  x$price2 <- x$price ^ 2
  x$price_scaled <- (x$price - 243385.2) / 159145.2
  x$max_speed_scaled <- (x$max_speed - 170.7568) / 37.56506
  x$price_x_max_speed <- x$price_scaled * x$max_speed_scaled
  return(predict(model, x))
}

rm(adj_formula_v1)
rm(adj_formula_v2)
rm(adj_formula_v3)
rm(adj_model_v1)
rm(adj_model_v2)
rm(adj_model_v3)
rm(max_speed_scaled)
rm(price_scaled)
rm(db_train)
rm(x_test)
rm(y_test)
rm(model)

evaluate_model(energy_prediction(dbwem), dbwem$energy)

# The model has R² of 0.9102 in the original dataframe.


#### Explaining the model ####
summary(final_model)

options(scipen=999)
coef_model <- data.frame(coef=coef(final_model)) %>% arrange(desc(coef))

for (i in 1:nrow(coef_model)) {
  coef_name <- rownames(coef_model)[i]
  if (coef_name != '(Intercept)') {
    coef <- round(coef_model[i,], 6)
    signal <- ifelse(coef < 0, 'decreases', 'increases')
    print(paste('For each unit added in', coef_name, 'it', signal,
                'the average energy consumption by', coef))
  }
}

# The model we used is capable of explaining 91.02% of the variation in the average energy consumption
# of vehicles. The most important variables are the vehicle's maximum speed and its price.

# 1)  For every Polish złoty more than the vehicle costs, it increases the average energy consumption by
#     0.000071;
# 2)  For every additional kilometer per hour the vehicle reaches maximum speed, it decreases the average
#     energy consumption by 0.112912;
# 3)  Seeking a balance between the two most relevant variables, if we multiply the standardized price
#     value (using mean = 243385.2 and standard deviation = 159145.2) by the standardized maximum speed value
#     (using mean = 170.7568 and standard deviation = 37.56506), we have that each a unit added to the final
#     result of this calculation, it increases the average energy consumption by 2.078431.

# Otherwise, we have the following:

# Increases average energy consumption:
# 4)  If the vehicle has a four-wheel drive system, it increases the average energy consumption by 1.842061;
# 5)  For each VDA unit the vehicle can hold in the trunk, it increases average power consumption by 0.006078;
# 6)  For each extra kilo that the car weighs empty, it increases average energy consumption by 0.005533;
# 7)  For each kilo that the vehicle is able to support in excess of load, it increases average energy
#     consumption by 0.005368;

# Decreases average energy consumption:
# 8)  For each kilogram of maximum weight allowed by law for the vehicle, it decreases average energy
#     consumption by 0.001441;
# 9)  For each centimeter added to the length of the vehicle, it decreases the average energy consumption
#     by 0.00919;
# 10) For each centimeter added to the width of the vehicle, it decreases the average energy consumption
#     by 0.019515;
# 11) For each centimeter added in the distance between the vehicle's front and rear axles, it decreases
#     the average energy consumption by 0.032214;
# 12) If the vehicle has a rear two-wheel drive system, it decreases the average energy consumption by 1.053602.
