#夢幻報表
# TODO: compare the 夢幻報表 with Logit Model

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import pymssql
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, label_ranking_average_precision_score

# Importing Data
from data import Dream_report, df

# Import data logic model revenue prediction
from main import overall_predicted_labels


# Dream Report predicted direction
dream_direct = Dream_report[['ddate', 'stockid', 'direction']]
#print(dream_direct)


# Import training data of logit model
ddate_column = df['ddate'].tolist()
stock_column = df['stockid'].tolist()

logit_direction = pd.DataFrame({'ddate': ddate_column,'stockid':stock_column, 'predict_labels': overall_predicted_labels})
#print(logit_direction)


# TODO: Compare the difference between Dream report & logit_direction
# Goal: 計算夢幻報表及 Logit Model 預測同向之數目(個股維度，分月份)


# Goal: 計算各月份夢幻報表及 Logit model 預測 0&1 的個數

# Goal: 預測數量相反者

# Goal: calculate the accuracy of Dream model

# print(len(dream_direct)) #183246
# print(len(logit_direction)) #201965

# To comfirm the dataframs are sorted in order

