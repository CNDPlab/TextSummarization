from Predictor.Utils.Scoring import Rouge
import numpy as np


rouge = Rouge()

def batch_scorer(list_pred_token_list, list_answer_token_list):
    scores = []
    for i in zip(list_pred_token_list, list_answer_token_list):
        precision, recall, f_score = rouge.rouge_l(i[0], i[1])
        scores.append(f_score)

    return np.mean(scores)