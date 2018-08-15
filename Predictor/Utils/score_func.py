from Predictor.Utils.Scoring import Rouge
import numpy as np
import torch as t
import ipdb


rouge = Rouge()

def cutting_mask(input_list, eos_id):
    nlist = []
    for i in input_list[1:]:
        if i != eos_id:
            nlist.append(i)
        else:
            break
    return nlist

def batch_scorer(list_pred_token_list, list_answer_token_list, eos_id):
    if isinstance(list_pred_token_list, t.Tensor):
        list_pred_token_list = list_pred_token_list.data.cpu().numpy()
    if isinstance(list_answer_token_list, t.Tensor):
        list_answer_token_list = list_answer_token_list.data.cpu().numpy()
    scores = []
    for i in zip(list_pred_token_list, list_answer_token_list):

        pre = [cutting_mask(i[0], '<EOS>')]
        tru = [cutting_mask(i[1], '<EOS>')]
        print(pre)
        print(tru)
        precision, recall, f_score = rouge.rouge_l(pre, tru)
        scores.append(f_score)
    return np.mean(scores)

if __name__ == '__main__':
    #TODO add test case

    input = [['<BOS>', '北京', '店', '店', '店', '店', '店', '店', '店', '店', '店', '店', '店', '店', '店', '店']]
    tru = [['<BOS>', '七月', '巨献', '#', '台', '特价', '卡罗拉', '抢购', '<EOS>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']]

#    pred = [['<BOS>', '长沙', '长沙', '#', '#', '受伤', '受伤', '受伤', '受伤', '受伤', '<EOS>', '<EOS>', '<EOS>', '<EOS>', '<EOS>', '<EOS>']]
    print(cutting_mask(input,'<EOS>'))
    print(cutting_mask(pred,'<EOS>'))
    print('------')
    print(batch_scorer(input, tru, '<EOS>'))
