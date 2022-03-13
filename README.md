# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project takes the development of a Machine Learning prototype from Jupyter Notebook and then scripted it following best practices to bring it to production.

The data corresponds to clients of a bank and the determination of whether they will be able to generate an expected return for the company, following personal attributes such as (income level, education, age, etc.)

Finally, Random Forest and logistic regression models are applied, it is determined that the best model is the first one, as expected.


## Running Files
to install the project, it is recommended to first clone to your local pc and create a virtual environment in Windows

```
python3 -m venv .env
```
or MacOS
```
virtualenv .env
```

Then you can activate in Windows
```
.env\Scripts\activate.bat
```
or MacOS
```
source .env/bin/activate
```

Next, install the requirements.txt files:

```
 pip install -r requirements.txt 
```

Finally, to make sure everything is in order, run the tests using the following code

```
ipython churn_script_logging_and_tests_solution.py
```

You can check in the logs folder, that all the tests passed successfully

```
root - INFO - Testing import_data: SUCCESS
root - INFO - Testing eda files: SUCCESS
root - INFO - Testing encoder helper: SUCCESS
root - INFO - Testing feature engineering: SUCCESS
root - INFO - Testing train:models: SUCCESS

```
