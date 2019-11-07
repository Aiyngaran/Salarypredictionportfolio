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
```
```
test_features_df.head(10)
```
```
train_features_df.shape
```
```
train_salaries_df.shape
```
```
test_features_df.shape
```
## Checking for the data type and counts of each Feature

```
train_features_df.info()
```
This tells the data types of each feature. There are no missing values as every entry is a non-null entry.
```
train_salaries_df.info()
```
There are no missing entities in the response as well.
```
test_features_df.info()
```
### Checking for the duplicates
```
train_features_df.duplicated().sum()
train_salaries_df.duplicated().sum()
test_features_df.duplicated().sum()
```
It turns out that there are no duplicates as it returned 0 for every command

### Separating Categorical and Numerical Columns
```
categorical_cols = ['jobId', 'companyId', 'jobType', 'degree', 'major', 'industry']
numeric_cols = ['yearsExperience', 'milesFromMetropolis']
```
### Basic Description of the Quantitative Features
```
train_features_df.describe()
```

### Basic Description of the Categorical Features
```
train_features_df.describe(include = ['O'])
```

### Merging Features and Response
```
train_df_eda = pd.merge(train_features_df, train_salaries_df, on = 'jobId', how = 'left')
```

### Visual examination of the response variable
Checking the distribution of the response variable
```
sns.set_style('darkgrid')
sns.distplot(train_df_eda.salary)
```

The Distribution seems to be slightly right skewed, might have some possible outliers. There are also some values around 0 salary.

Box plot to check outliers
```
train_df_eda.boxplot(column = 'salary')
```
There seems to be outliers on bot the sides of the plot and there are more outliers above the upper limit of the whiskers plot than below the lower limit of the whiskers plot

```
stat = train_df_eda.salary.describe()
IQR = stat['75%'] - stat['25%']
upper_limit = stat['75%'] + 1.5 * IQR
lower_limit = stat['25%'] - 1.5 * IQR
print('The upper limit is ', upper_limit)
print('The lower limit is ', lower_limit)
```
The upper limit is  220.5
The lower limit is  8.5

Deeper examination of outliers
```
 train_df_eda[train_df_eda['salary'] <= lower_limit]
```
The salaries with 0 values does not seem like actual entries, they should be missing values and these potential outliers have to be removed

```
train_df_eda[train_df_eda['salary'] >= upper_limit].head(10)
```
The higher values in salary are ought to the job type or the degree or the industry. So they are not the outliers and are retained for the analysis but the job type JUNIOR has to be examined
```
train_df_eda[(train_df_eda['salary'] >= upper_limit) & (train_df_eda['jobType'] == 'JUNIOR')].head(10)
```
The JUNIOR role is being paid more in industries like oil and finance which is reasonable

### Removing the data points with zero salaries
```
train_df_eda = train_df_eda[train_df_eda['salary'] > lower_limit]
```

### Visual examination of the features
```
def plot_numeric(df, col):
    df.plot(kind = 'scatter', x = col, y = 'salary', alpha = 0.1)
for column in ['yearsExperience', 'milesFromMetropolis']:
    plot_numeric(train_df_eda, column)
```
As years of experience increases, the salary tends to increase and the as the mile from metropolis increases, the salary tends to decrease.

```
def plot_features(df, col):
    plt.figure(figsize = (14,6))

    if df[col].dtype == 'int64':
        mean = df.groupby(col)['salary'].mean()
        std = df.groupby(col)['salary'].std()
        plt.fill_between(range(len(std)), mean.values - std.values, mean.values + std.values, alpha = 0.1)
        mean.plot()
    elif col == 'companyId':
        df[col] = df[col].astype('category')
        mean = df.groupby(col)['salary'].mean()
        std = df.groupby(col)['salary'].std()
        plt.fill_between(range(len(std)), mean.values - std.values, mean.values + std.values, alpha = 0.1)
        mean.plot()
    else:
        median = df.groupby(col)['salary'].median()
        df[col] = df[col].astype('category')
        levels = median.sort_values().index.tolist()
        df[col].cat.reorder_categories(levels, inplace = True)
        sns.boxplot(x = col, y = 'salary', data = df)
    plt.xticks(rotation = 45)
    plt.xlabel(col)
    plt.ylabel('Salary')
    plt.show()
```
```
for col in ['companyId', 'jobType', 'degree', 'major', 'industry', 'yearsExperience', 'milesFromMetropolis']:
    plot_features(train_df_eda, col)
```
From the plot for company ID and Salary it is clear that companyID doesnt have any particular linear relationship with the salary on itself. The job type, degree, industry and years experience has a significant positive linear relationship with salary and miles from metropolis has a clear negative linear relationship with salary. The major seems to have some kind of relationship with salary but not very significant.

### Converting categorical variables to numerical to check correlation
```
def encode_labels(df, col):
    cat_dict = {}
    cats = df[col].cat.categories.tolist()
    for cat in cats:
        cat_dict[cat] = train_df_eda[train_df_eda[col] == cat]['salary'].mean()
    df[col] = df[col].map(cat_dict)
```
```
for col in train_df_eda.columns:
    if train_df_eda[col].dtype.name == 'category':
        encode_labels(train_df_eda, col)
```
Every categorical variable has been converted into numerical variable by taking salary mean for the respective category.

### Correlation Plot
```
plt.figure(figsize = (13,10))
features = ['companyId', 'jobType', 'degree', 'major', 'industry',
       'yearsExperience', 'milesFromMetropolis', 'salary']
correlation = train_df_eda[features].corr()
sns.heatmap(correlation, cmap = 'Blues', annot = True)
plt.xticks(rotation = 90)
plt.show()
```
From the correlation plot it can be seen that there are correlations between the target variable (salary) and every other except for the companyID. There are also some inter relations between the features like major, degree, and job type.

## Defining the required functin to reuse
```
def combine_df(df1, df2):
    return pd.concat([df1, df2], axis = 1, sort = False)

def clean_df(df):
    clean_1 = df.drop('jobId', axis = 1)
    if 'salary' in df.columns:
        clean_2 = clean_1[clean_1['salary'] > 0]
        return clean_2
    else:
        return clean_1

def one_hot_encode(df):
    return pd.get_dummies(df)

def get_features(df, target):
    return df.loc[:, df.columns != target]

def get_target(df, target):
    return df[target]

def train_model(features, target, model_object):
    model_scores = cross_val_score(model_object, features, target, scoring = 'neg_mean_squared_error', cv = 5)
    mse = -model_scores.mean()
    return (mse)

def save_model(model, feature_importance, predictions):
    pickle.dump(model, open('final_model_salary_prediction', 'wb'))
    feature_importance.to_csv('Salary_prediction_feature_importance.csv')
    np.savetxt('Salary_predictions.csv', predictions, delimiter = ',')
```
- combine_df function combines the two dataframes, here it is hte features and the target.
- clean_df function initially drops the jobID column as is does not make sense to have it for prediction and then it drops all the rows where the salary is equal to 0 as we havw some salaries equal to zero.
- one_hot_encode function converts all the columns with categorical variables into separate individual columns.
- get_features function extracts all the features from the combined dataframe.
- get_target function extracts the target variable from the combined dataframe.
- train_model function trains the model specified.
- save_model function saves the model, feature importance and the predictions for deployment
