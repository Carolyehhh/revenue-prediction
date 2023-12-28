# Logit Model

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
from data import df, X, y, target_col, variable_col, ym

# Iporting Functions
from module1 import predict_for_month, find_best_threshold, predicted_labels, predicted_accuracy



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


# TODO: 計算模型各月的預測準確度
# 初始化結果列表
overall_prob = []
overall_predicted_labels = []
overall_accuracy = []

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

    # calculate model acuracy
    mdl_accuracy = predicted_accuracy(monthly_data[target_col], pred_label)

    # 將結果串聯到整體結果列表中
    overall_accuracy.append(mdl_accuracy)


#print(np.mean(overall_accuracy)) #0.63
#print(len(overall_predicted_labels)) #201965
#print(ym) #202310-202309-202308


