<a href="https://github.com/WallPasq/projects" rel="noreferrer noopener" target="_blank">
  <img src="https://github.com/WallPasq/projects/blob/main/assets/banner.png" alt="Languages ​​and frameworks icons: Python, scikit-learn, Spark, Databricks, BigQuery, Streamlit, Airflow, PowerBI. Projects." rel="noreferrer noopener" target="_blank">
</a>
<p>It's my portfolio right now. There are projects here that I have been working on in my studies, courses, inspirations, and other reasons.</p>

<h1>Folders</h1>
<p>Each project is in a different folder, and here are brief descriptions about them.</p>

<h2>
    <a href="https://www.datascienceacademy.com.br/course/analise-de-dados-com-r">
        Big Data Analytics with R and Microsoft Azure Machine Learning
    </a>
</h2>
<p>Projects developed during the Data Science Academy course.</p>

<h3>
    <a href="https://github.com/WallPasq/projects/tree/83bcd63f7a5875129c2b468defd7c48672d896c8/world_happiness_report_data">
        World Happiness Report Data
    </a>
</h3>
<p>The analysis of socioeconomic data is essential for any company to understand the social and economic scenario for the business environment, understand the difference between regions, cities and countries, the influence of indicators on customers purchasing decisions and how changes in the scenario can impact corporate strategies.</p>
<p>In this project I analyzed real data of socioeconomic indicators from different countries, in a time series. The most important work was the data munging, where I used imputation techniques such as Last and Next Observation Carried Forward (LOCF and NOCF) and replacement by the average data of the region of the year. Using the dplyr and ggplot2 libraries, I answered five questions through the data.</p>

<h3>
    <a href="https://github.com/WallPasq/projects/tree/83bcd63f7a5875129c2b468defd7c48672d896c8/health_financial_data_analysis">
        Health Financial Data Analysis
    </a>
</h3>
<p>Using data from a national hospital cost survey conducted by the USAgency for Healthcare consisting of hospital records of inpatient samples, I answered 17 business questions using SQL and R language.</p>
<p>Among the techniques used to answer the questions are: ANOVA test and linear regression analysis.</p>
<p>The data provided are restricted to the city of Wisconsin and refer to patients aged 0 to 17 years.</p>

<h3>
    <a href="https://github.com/WallPasq/projects/tree/d7ac99fe947d8e7a51aa76e86c072c9b0a800f42/industrial_equipment_maintenance">
        Industrial Equipment Maintenance
    </a>
</h3>
<p>Using a mass of dummy data created in code, to have a controlled environment, I show my skills by explaining not only how the variables arrived at the result in the model, but also why. For this, I use the h2o library to create a series of Machine Learning models using AutoML and, from the best model, I analyze the SHAP values ​​of each variable.</p>
<p>I answer only one, but extremely important, business question: What factors (metrics) most contribute to explain the behavior of the need for maintenance in a piece of equipment? Why?</p>

<h3>
    <a href="https://github.com/WallPasq/projects/tree/d356c5f54415b11a4f825a871b99af6607684898/energy_consumption_forecasting">
        Energy Consumption Forecasting
    </a>
</h3>
<p>Using the dataset "Dataset of electric passenger cars with their specifications", which contains publicly available real data, I built a Machine Learning model capable of predicting the energy consumption of electric vehicles based on several factors, such as price, the maximum speed of the vehicle, its weight, its load capacity, among other attributes.</p>
<p>This is my first project with real project characteristics: at the beginning, I insert the context of the project, the business question, how to solve it, what is the definition of done, and a report on the data, with reference, summary, the link to the data source and a data dictionary made with the help of ChatGPT (since it was not provided with the dataset). Afterwards, I carry out all the necessary processes to execute the project, such as handling the missing values, exploratory analysis, modifying metadata, developing and evaluating the Machine Learning models, fine-tuning the model and interpreting the final model.</p>
<p>In this project, I used Linear Regression as a tool to treat missing data, and also to create the final model. In this way, I show how Linear Regression (and other algorithms) can be used both as statistical tools and as Machine Learning models. In addition, at the end, I explain how each variable behaves for the final prediction of the model. That is, I answer a possible business question: how does each characteristic of the vehicle affect its average energy consumption?</p>

<h3>
    <a href="https://github.com/WallPasq/projects/tree/625fe48581daff99ab5c0f3f9ec8e864239a3970/efficiency_fire_extinguishers">
        Efficiency Fire Extinguishers
    </a>
</h3>
<p>Using the dataset "Extinguishing status of fuel flames with sound wave", which contains a study about extinguishing tests of four different fuel flames with a sound wave extinguishing system, I built a Neural Network Machine Learning Model able to predict whether or not an extinguisher will be able to extinguish a flame. For this, it uses six variables, which are: flame size, type of fuel, distance from the flame to the collimator output, sound pressure level, airflow created by the sound wave and low frequency range.</p>
<p>The idea of the model is to use it in computer simulations, aiming to add a layer of security to the analysis of the efficiency of fire extinguishers. Some important points I learned from this project:</p>
<ol>
  <li>Before thinking about modifying the variables, if possible, use another ML model, as it can perform much better with the same training and testing dataset;</li>
  <li>The softmax function can be used as the last activation function of a neural network to normalize the output of a network to a probability distribution. With this distribution it is possible to calculate the ROC curve of the binomial model;</li>
  <li>Exploratory analysis is essential to understand the relationships between variables, which helps to explain complex ML models such as neural networks.</li>
</ol>
