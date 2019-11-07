# Salary prediction portfolio
salary prediction project in Python

Author:  **Aiyngaran Chokalingam**
---

## PART 1 DEFINE
### ---- 1 Define the problem ----

##### From an employer's perspective, if the firm employes a person for a particular position it has to offer a package whcih best suits the position in that location. It is most of the times extremely higher or lower than what is actual. So, this project intends to provie a model which can predict the salary provided the company, location, position and a few other inforamtion using which the company will be able to offer a salary with a known variablity. From an employee's perspective, he/she can choose a company based on their qualification and salary expectations or decide their career based on the it.

---

## Requirements
There are some general library requirements for the project and some of which are specific to individual methods. The general requirements are as follows.
- *numpy*
- *pandas*
- *matplotlib*
- *pickle*
- *sklearn*
- *seaborn*

**NOTE**:  I have used Anaconda distribution of Python, but any IDE can be used to execute the code.
---

## Explanation of the Code

The code, `salaryprediction.ipynb`, begins by importing necessary Python packages:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression #model_1
from sklearn.tree import DecisionTreeRegressor #model_2
from sklearn.ensemble import RandomForestRegressor #model_3
from sklearn.ensemble import AdaBoostRegressor #model_4
from sklearn.ensemble import GradientBoostingRegressor #model_5
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import pickle
```

- *NOTE:  use pip install "name of the library" in anaconda prompt to install the library files which are not predefined in python. Example pip install nltk*

## PART 2 DISCOVER
### ---- 2 Load the data ----

Loading all the necessary datasets
```
def load_df(path):
    return pd.read_csv(path)
train_features_df = load_df('train_features.csv')
train_salaries_df = load_df('train_salaries.csv')
test_features_df = load_df('test_features.csv')
```

### ---- 3 Explore the data (EDA) ----

We initially examine how the data set looks like by looking at the first 10 rows and its dimensions
```
train_features_df.head(10)
```
```
train_salaries_df.head(10)
test_features_df.head(10)
train_features_df.shape
train_salaries_df.shape
test_features_df.shape
```
