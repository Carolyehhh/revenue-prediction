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

# Import data logic model revenue prediction
from revenue_prediction_main import overall_predicted_labels


#================== Data Cleaning ===========================================
conn = pymssql.connect('192.168.121.50', 'carol_yeh', 'Cmoneywork1102', 'master')

cursor = conn.cursor()

qr_sub = f""" 
    SELECT [CTIME]
      ,[MTIME]
      ,[年月]
      ,[股票代號]
      ,[營收月變動]
      ,[營收年成長]
      ,[月營收信心度]
      ,[預估單月合併營收創新高]
      ,[預估單月合併營收年成長(%)]
     
     from [DataFinance].[dbo].[ForcastMonthRevenue_Result]

     order by 年月 desc
     """ 
Dream_report = pd.read_sql(qr_sub, conn)
#print(Dream_report.head())

# change column names into English
Dream_report.rename(columns={'年月': 'ddate', '股票代號':'stockid', '營收月變動':'direction'}, inplace=True)

# Dream Report predicted direction
dream_direct = Dream_report[['ddate', 'stockid', 'direction']]
#print(dream_direct)


# Import training data of logit model
logit_df = pd.read_csv('C:/Users/user1/Desktop/Cmoney/上市櫃營收預測/revrawdata.csv')

# Drop NaNs from data
logit_df = logit_df.dropna()
#print(type(df))
nan_check = logit_df.isna().sum()
#print(nan_check)

# Logit Model predicted direction
ddate_column = logit_df['ddate'].tolist()
stock_column = logit_df['stockid'].tolist()

logit_direction = pd.DataFrame({'ddate': ddate_column,'stockid':stock_column, 'predict_labels': overall_predicted_labels})
print(logit_direction)
