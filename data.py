import numpy as np
import pandas as pd
import pymssql

#================== Logit Model Data Cleaning ===========================================
# Import training data
df = pd.read_csv('C:/Users/user1/Desktop/Cmoney/上市櫃營收預測/revrawdata.csv')

# Drop NaNs from data
df = df.dropna()
#print(type(df))
nan_check = df.isna().sum()
#print(nan_check)

unique_stockid = df['stockid'].unique()
ym = df['ddate'].unique()
#print(len(unique_stockid)) #1830

target_col = ['direction']
variable_col =['revd1', 'yoyd1',
                         'mond1', 'revd2', 'yoyd2', 'mond2', 'revd3', 'yoyd3', 'mond3', 'revd4',
                         'yoyd4', 'mond4', 'revd5', 'yoyd5', 'mond5', 'revd6', 'yoyd6', 'mond6',
                         'revd7', 'yoyd7', 'mond7', 'revd8', 'yoyd8', 'mond8', 'revd9', 'yoyd9',
                         'mond9', 'revd10', 'yoyd10', 'mond10', 'revd11', 'yoyd11', 'mond11',
                         'revd12', 'yoyd12', 'mond12']

# Prepare data for training
y = df[target_col] #overall y
X = df[variable_col] # overall X



#================== Logit Model Data Cleaning ===========================================
conn = pymssql.connect('192.168.121.50', 'carol_yeh', 'Cmoneywork1102', 'master')

cursor = conn.cursor()

qr_sub = f""" 
    SELECT 
      [年月]
      ,[股票代號]
      ,[月變動方向]
     
     from [DataFinance].[dbo].[ForcastMonthRevenue]

     order by 年月 desc
     """ 
Dream_report = pd.read_sql(qr_sub, conn)


# change column names into English
Dream_report.rename(columns={'年月': 'ddate', '股票代號':'stockid', '月變動方向':'direction'}, inplace=True)
#print(Dream_report.head())

