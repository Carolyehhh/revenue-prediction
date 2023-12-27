# TODO: probability prediction by months

def best_threshold(original_direction, probability):

    """
    Goal: find the best threshold for determination
    Output: A float
    Original_direction: 實際結果
    Probability: 由 predict_for_month 預測的營收上漲機率
    """
    
    # 計算閾值
    fpr, tpr, thresholds = roc_curve(original_direction, probability)

    # to find the threshold that maximizes Youden's Index
    youden_index = tpr - fpr
    optimal_treshold_idx = np.argmax(youden_index)

    return thresholds[optimal_treshold_idx]



def predict_for_month(model_fit, dataset, variable_col, best_threshold):

    """
    Output: probability, predicted_labels
    model_fit: the fitted model
    dataset: data to be prediced dataframe
    variable_col: list of variable names
    best_threshold: float of threshold
    """

    # 提取特徵X
    X_monthly = dataset[variable_col]

    # 預測蓋率
    probability = model_fit.predict_proba(X_monthly)[:1]

    # 二元預測標籤
    predicted_labels = (probability > best_threshold).astype(int)

    return probability, predicted_labels


print("HIII")