import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
# 用於二元目標變數
from scipy.stats import pointbiserialr
from functools import reduce
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

from sklearn.feature_selection import RFE

from GDP_train.ipynb import X_train_MacroRev

model_MacroRev = LogisticRegression()
model_fit_MacroRev = model_MacroRev.fit(X_train_MacroRev, y_train_MacroRev)

# 針對總經+12月營收判斷
# 使用 RFE 進行變數選擇
selector = RFE(model_MacroRev, n_features_to_select=20) #留下20 個 features
selector = selector.fit(X_MacroRev, y_MacroRev)

# Get the ranking of each feature
featre_ranking = selector.ranking_

featre_ranking
# 輸出選擇的特徵
selected_features = X_MacroRev.columns[selector.support_]
#print(selected_features)