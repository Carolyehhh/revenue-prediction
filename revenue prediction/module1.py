# TODO: probability prediction by months

def best_threshold():

    """
    Goal: find the best threshold for determination
    Output: A float
    
    """
    raise TypeError("Unfinished function.")

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