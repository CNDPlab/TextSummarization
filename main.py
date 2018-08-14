from configs import Config
from torch.utils.data import DataLoader
import torch as t
import fire
from Predictor.Utils import Vocab
import pickle as pk
from DataSets import DataSet, own_collate_fn
from Predictor.Utils.loss import masked_cross_entropy
from Predictor.Utils import batch_scorer
from Trainner import Trainner
from Predictor import Models
import os
import ipdb


def train(**kwargs):
    args = Config()
    args.parse(kwargs)
    loss_func = masked_cross_entropy
    score_func = batch_scorer
    train_set = DataSet('processed/train/')
    dev_set = DataSet('processed/dev/')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
    vocab = pk.load(open('Predictor/Utils/vocab.pkl', 'rb'))
    eos_id, sos_id = vocab.token2id['<EOS>'], vocab.token2id['<BOS>']
    args.eos_id = eos_id
    args.sos_id = sos_id
    model = getattr(Models, args.model_name)(matrix=vocab.matrix, args=args)
    trainner = Trainner(args)
    trainner.train(model, loss_func, score_func, train_loader, dev_loader, teacher_forcing_ratio=0.5, resume=args.resume)

def select_best_model(save_path):
    file_name = os.listdir(save_path)
    best_model = sorted(file_name, key = lambda x: x.split('_')[1], reverse=True)[0]
    return best_model

def test(**kwargs):
    args = Config()
    test_set = DataSet('processed/test/')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
    vocab = pk.load(open('Predictor/Utils/vocab.pkl', 'rb'))
    model = getattr(Models, args.model_name)(matrix=vocab.matrix, args=args)
    model.load_state_dicts(t.load())
    while True:
        x = input('input context:')
        token_x = predict_pipeline(x)
        lenth_x = len(token_x)
        input_context = t.Tensor([token_x]).long()
        input_context_lenth = t.Tensor([lenth_x]).long()
    #TODO complete load_state_dict and predict
    model.load_state_dicts(t.load(select_best_model(args.saved_model_root)))
    model.use_teacher_forcing = False
    for data in test_loader:
        pass


    # while True:
    #     x = input('input context:')
    #     token_x = predict_pipeline(x)
    #     lenth_x = len(token_x)
    #     input_context = t.Tensor([token_x]).long()
    #     input_context_lenth = t.Tensor([lenth_x]).long()



# def predict(**kwargs):
#     args = Config()
#
#     model = getattr(Models, args.model_name)(matrix=vocab.matrix, args=args)
#     model.load_state_dicts(t.load())


if __name__ == '__main__':
    fire.Fire()


