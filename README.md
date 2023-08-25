# Projects
It's my portfolio right now. There are projects here that I have been working in my studies, courses, inspirations, and other reasons.

# Folders
Each project is in a different folder, and here are brief descriptions about them.

## Projects developed during the [Big Data Analytics with R and Microsoft Azure Machine Learning](https://www.datascienceacademy.com.br/course/analise-de-dados-com-r) course, from the Data Science Academy

### [World Happiness Report Data](https://github.com/WallPasq/projects/tree/83bcd63f7a5875129c2b468defd7c48672d896c8/world_happiness_report_data)

The analysis of socioeconomic data is essential for any company to understand the social and economic scenario for the business environment, understand the difference between regions, cities and countries, the influence of indicators on customers purchasing decisions and how changes in the scenario can impact corporate strategies.

In this project I analyzed real data of socioeconomic indicators from different countries, in a time series. The most important work was the data munging, where I used imputation techniques such as Last and Next Observation Carried Forward (LOCF and NOCF) and replacement by the average data of the region of the year. Using the dplyr and ggplot2 libraries, I answered five questions through the data.

### [Health Financial Data Analysis](https://github.com/WallPasq/projects/tree/83bcd63f7a5875129c2b468defd7c48672d896c8/health_financial_data_analysis)

Using data from a national hospital cost survey conducted by the USAgency for Healthcare consisting of hospital records of inpatient samples, I answered 17 business questions using SQL and R language.
Among the techniques used to answer the questions are: ANOVA test and linear regression analysis.

The data provided are restricted to the city of Wisconsin and refer to patients aged 0 to 17 years.

### [Industrial Equipment Maintenance](https://github.com/WallPasq/projects/tree/d7ac99fe947d8e7a51aa76e86c072c9b0a800f42/industrial_equipment_maintenance)

Using a mass of dummy data created in code, to have a controlled environment, I show my skills by explaining not only how the variables arrived at the result in the model, but also why. For this, I use the h2o library to create a series of Machine Learning models using AutoML and, from the best model, I analyze the SHAP values ​​of each variable.

I answer only one, but extremely important, business question: What factors (metrics) most contribute to explain the behavior of the need for maintenance in a piece of equipment? Why?

### [Energy Consumption Forecasting](https://github.com/WallPasq/projects/tree/d356c5f54415b11a4f825a871b99af6607684898/energy_consumption_forecasting)

Using the dataset "Dataset of electric passenger cars with their specifications", which contains publicly available real data, I built a Machine Learning model capable of predicting the energy consumption of electric vehicles based on several factors, such as price, the maximum speed of the vehicle, its weight, its load capacity, among other attributes.

This is my first project with real project characteristics: at the beginning, I insert the context of the project, the business problem, how to solve it, what is the definition of complete, and a report on the data, with reference, summary, the link to the data source and a data dictionary made with the help of ChatGPT (since it was not provided with the dataset). Afterwards, I carry out all the necessary processes to understand the relationships between the data, such as handling the missing values, exploratory analysis, modifying metadata, developing and evaluating the models, fine-tuning the models and interpreting the final model.

In this project, I used Linear Regression as a tool to treat missing data, and also to create the final model. In this way, I show how Linear Regression (and other algorithms) can be used both as statistical tools and as Machine Learning models. In addition, at the end, I explain how each variable behaves for the final prediction of the model. That is, I answer a possible business question: how does each characteristic of the vehicle affect its average energy consumption?
