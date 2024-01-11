from sklearn.metrics import roc_curve
import numpy as np
from sklearn.metrics import accuracy_score

# TODO: probability prediction by months


def find_best_threshold(original_direction, predicted_probability):

    """
    Goal: find the best threshold for determination
    Output: A float
    Original_direction: 實際結果
    Probability: 由 predict_for_month 預測的營收上漲機率
    """

    # 計算閾值以及fpr、tpr
    fpr, tpr, thresholds = roc_curve(original_direction, predicted_probability)

    # to find the threshold that maximizes Youden's Index
    youden_index = tpr - fpr
    # 找最大的 youden_index 做為最佳的閾值index
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


def predicted_labels(pred_probability, optimal_treshold):
    """
    Output: 以閾值為界判斷預測值應為0或1
    pred_probability: probability prediced by 'predict_for_month'
    optimal_treshold: optimal threshold calculated by
                      function 'find_best_threshold'
    """
    # 二元預測標籤: 將大於閾值的預測機率標註為1(上漲)，其餘則標註為0(下跌)
    predicted_label_results = (pred_probability > optimal_treshold).astype(int)

    return predicted_label_results


def predicted_accuracy(original_direction, predicted_label_results):
    """
    Output: model_accuracy
    Original_direction: directions(Y) from the rqw data
    Predicted_label_results: predicted label from 'predicted_labels'
    """
    model_accuracy = round(accuracy_score(
        original_direction, predicted_label_results), 2)

    return model_accuracy

# class

# 比對預測結果相同
def compare_predictions(group):
    """
    Outcome: Determine whether the prediction of the model is correct
    group:A dataframe groupby specific variables contains the predictions of 
          both Dream Report & Logit Model, also the real direction 
    """
    group['DreamPredCorrect'] = (group['direction'] == group['direction_real']).astype(int)
    group['LogitPredCorrect'] = (group['predicted_labels'] == 
                                 group['direction_real']).astype(int)

    return group

