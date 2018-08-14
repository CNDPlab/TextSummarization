from Predictor.Utils.Scoring import Rouge
import numpy as np
import torch as t
import ipdb


rouge = Rouge()

def cutting_mask(input_list, eos_id):
    nlist = []
    for i in input_list:
        if i != eos_id:
            nlist.append(i)
        else:
            nlist.append(i)
            break
    return nlist

def batch_scorer(list_pred_token_list, list_answer_token_list, eos_id):

    if isinstance(list_pred_token_list, t.Tensor):
        list_pred_token_list = list_pred_token_list.data.cpu().numpy()
    if isinstance(list_answer_token_list, t.Tensor):
        list_answer_token_list = list_answer_token_list.data.cpu().numpy()
    scores = []
    for i in zip(list_pred_token_list, list_answer_token_list):

        precision, recall, f_score = rouge.rouge_l([cutting_mask(i[0], eos_id)], [cutting_mask(i[1][1:], eos_id)])
        scores.append(f_score)

    return np.mean(scores)

if __name__ == '__main__':
    input = t.Tensor([[1, 2, 3], [2, 3, 4]]).long()
    pred = t.Tensor([[1, 2, 3], [2, 3, 4]]).long()
    print(batch_scorer(input, pred))
