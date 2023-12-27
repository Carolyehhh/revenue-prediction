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
from sklearn.metrics import f1_score, recall_score, precision_score

# Import data
df = pd.read_csv('C:/Users/user1/Desktop/Cmoney/上市櫃營收預測/revrawdata.csv')
print(df)



