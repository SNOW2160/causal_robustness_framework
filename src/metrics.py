import numpy as np

def calculate_pehe(true_cate, pred_cate):
    return np.sqrt(np.mean((true_cate - pred_cate)**2))

def calculate_pss(pred_cate_placebo):
    """
    Placebo Sensitivity Score (PSS).
    Input: Predicted CATE on Placebo data (where True CATE = 0).
    Output: Mean Absolute Hallucination.
    """
    return np.mean(np.abs(pred_cate_placebo))