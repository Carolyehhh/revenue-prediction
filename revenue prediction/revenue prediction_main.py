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

# Iporting Functions
from module1 import predict_for_month, find_best_threshold, predicted_labels

#================== Data Cleaning ===========================================
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

#================================= Main ===========================================

# TODO: Model Training
# Split data by TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=50)

# each split contains train set & test set index
splits = list(tscv.split(df))

# 使用最後一組的 trained model index 切分 training & testing data
train_idx = splits[-1][0]
test_idx = splits[-1][1]

# Extract test and train data from overall data
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Train model based on the training data
model = LogisticRegression()
model_fit = model.fit(X_train, y_train)

# TODO: probability prediction by months
# 初始化結果列表
overall_prob = []
overall_predicted_labels = []

# 迴圈中調用函數
for date in ym:
    # 提取特定月份的資料
    monthly_data = df[df['ddate'] == date].copy()

     # 使用函數進行預測
    probability = predict_for_month(model_fit, monthly_data, variable_col)

    # 使用best_threshold函數找到最佳閾值
    threshold_for_date = find_best_threshold(monthly_data[target_col], probability)

    # 貼方向標籤
    #print(threshold_for_date)
    pred_label = predicted_labels(probability, threshold_for_date)

    # 將結果串聯到整體結果列表中
    overall_prob.extend(probability)
    overall_predicted_labels.extend(pred_label)

print(len(overall_prob))
print(len(overall_predicted_labels))

