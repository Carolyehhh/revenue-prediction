from sklearn.metrics import roc_curve, auc
import numpy as np
# TODO: probability prediction by months


def find_best_threshold(original_direction, predicted_probability):

    """
    Goal: find the best threshold for determination
    Output: A float
    Original_direction: 實際結果
    Probability: 由 predict_for_month 預測的營收上漲機率
    """
    
    # 計算閾值
    fpr, tpr, thresholds = roc_curve(original_direction, predicted_probability)

    # to find the threshold that maximizes Youden's Index
    youden_index = tpr - fpr
    optimal_threshold_idx = np.argmax(youden_index)

    return thresholds[optimal_threshold_idx]



def predict_for_month(model_fit, dataset, variable_col):

    """
    Output: probability, predicted_labels
    model_fit: the fitted model
    dataset: data to be prediced dataframe
    variable_col: list of variable names
    """

    # 提取特徵X
    X_monthly = dataset[variable_col]

    # 預測蓋率
    probability = model_fit.predict_proba(X_monthly)[:, 1]

    return probability


def predicted_labels(pred_probability,optimal_treshold):
    # 二元預測標籤
    predicted_labels = (pred_probability > optimal_treshold).astype(int)

    return predicted_labels
