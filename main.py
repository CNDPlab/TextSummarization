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
import ipdb


def train(**kwargs):
    args = Config()
    args.parse(kwargs)
    loss_func = masked_cross_entropy
    #TODO complete score_func
    score_func = batch_scorer
    train_set = DataSet('processed/train/')
    dev_set = DataSet('processed/dev/')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
    vocab = pk.load(open('Predictor/Utils/vocab.pkl', 'rb'))
    model = getattr(Models, args.model_name)(matrix=vocab.matrix, args=args)
    trainner = Trainner(args)
    trainner.train(model, loss_func, score_func, train_loader, dev_loader, teacher_forcing_ratio=1,resume=args.resume)

def test(**kwargs):
    args = Config()
    test_set = DataSet('processed/test/')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)


    model = getattr(Models, args.model_name)(matrix=vocab.matrix, args=args)
    #TODO complete load_state_dict and predict
    model.load_state_dicts()
    with t.no_grad():
        outputs = model()


if __name__ == '__main__':
    fire.Fire()


