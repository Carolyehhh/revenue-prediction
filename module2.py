# functions for the Dream Report

def identical_dream_logit(original_data ,specific_month):
    """
    Goal: 計算夢幻報表與 Logit Model 預測同方向之數目
    original_data: original data from Dream Report
    specific_month: 特定月份
    """
    # Extract specific month data
    dream_data = original_data.loc[original_data['ddate'] == specific_month]

    raise TypeError('Undone Fuction~')

    """
    for date in ym:
        identical_dream_logit(date)
    """