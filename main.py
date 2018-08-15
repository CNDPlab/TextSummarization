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
    train_set = DataSet(args.processed_folder+'train/')
    dev_set = DataSet(args.processed_folder+'dev/')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
    vocab = pk.load(open('Predictor/Utils/vocab.pkl', 'rb'))
    eos_id, sos_id = vocab.token2id['<EOS>'], vocab.token2id['<BOS>']
    args.eos_id = eos_id
    args.sos_id = sos_id
    model = getattr(Models, args.model_name)(matrix=vocab.matrix, args=args)
    trainner = Trainner(args, vocab)
    trainner.train(model, loss_func, score_func, train_loader, dev_loader, teacher_forcing_ratio=args.init_tf_ratio, resume=args.resume)

def select_best_model(save_path):
    file_name = os.listdir(save_path)
    best_model = sorted(file_name, key=lambda x: x.split('_')[1], reverse=True)[0]
    return best_model

def _load(path, model):
    resume_checkpoint = t.load(os.path.join(path, 'trainer_state'))
    model.load_state_dict(t.load(os.path.join(path, 'model')))
    return {'epoch': resume_checkpoint['epoch'],
            'step': resume_checkpoint['step'],
            'optimizer': resume_checkpoint['optimizer'],
            'model': model}

def test(**kwargs):
    args = Config()
    test_set = DataSet(args.processed_folder+'test/')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
    vocab = pk.load(open('Predictor/Utils/vocab.pkl', 'rb'))
    eos_id, sos_id = vocab.token2id['<EOS>'], vocab.token2id['<BOS>']
    args.eos_id = eos_id
    args.sos_id = sos_id
    model = getattr(Models, args.model_name)(matrix=vocab.matrix, args=args)
    load = _load('ckpt/saved_models/2018_08_14_04_33_40_0.32190511440828723', model)
    model = load['model']
    model.to('cuda')
    #TODO complete load_state_dict and predict
    model.teacher_forcing_ratio = -100
    with t.no_grad():
        for data in test_loader:
            context, title, context_lenths, title_lenths = [i.to('cuda') for i in data]
            token_id, prob_vector, token_lenth, attention_matrix = model(context, context_lenths, title)
            score = batch_scorer(token_id.tolist(),title.tolist(),args.eos_id)
            context_word = [[vocab.from_id_token(id.item()) for id in sample] for sample in context]
            words = [[vocab.from_id_token(id.item()) for id in sample] for sample in token_id]
            title_words = [[vocab.from_id_token(id.item()) for id in sample] for sample in title]
            for i in zip(context_word, words, title_words):
                a = input('next')
                print(f'context:{i[0]},pre:{i[1]}, tru:{i[2]}, score:{score}')


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


