# Data Science Coursework 2

This repository contains the code for the second coursework of the Data Science module. The coursework is divided into two parts. The first part is about the visualisation of the data and the second part is about the prediction of the data.

## Pre-requisites
- Data: The data for the coursework is provided in the `01871834-math70076-assessment-2-data` folder, which contains 2 sub-folders: `covid19-report`(for visualisation tasks) and `covid19-global-forecasting` (for forecasting task).
- Pakages: All the required packages are listed in the `requirements.txt` file. One can install all the required packages by running the following command:
```bash
pip install -r requirements.txt
```

## Visualisation
The visualization section of the coursework focuses on representing COVID-19 data graphically. This data is located in the `covid19-report` folder and has already been cleaned for this specific task. The visualization component consists of three distinct tasks:
- Task 1: Visualising the number of confirmed cases, deaths, recoveries and growth factor over time.
- Task 2: Visualising the number of confirmed cases, deaths, and recoveries per continent.
- Task 3: Visualising the number of confirmed cases, deaths, and recoveries for top 10 influenced country.

## Forecasting
The forcasting section of the coursework predicts the number of confirmed cases for the next 20 days. The data for this task is located in the `covid19-global-forecasting` folder. The forecasting component consists of 3 distinct tasks:
- Task 1: Using the SIR model to explain the spread of the virus.
- Task 2: Fit a linear regression model to the number of confirmed cases.
- Task 3: Fit a logistic regression model to predict the number of confirmed cases.