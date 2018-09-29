from configs import Config
from torch.utils.data import DataLoader
import torch as t
import fire
from Predictor.Utils import Vocab
import pickle as pk
from DataSets import DataSet, own_collate_fn
from Predictor.Utils.loss import masked_cross_entropy, mixed_loss, masked_cross_entropy2
from Predictor.Utils import batch_scorer
from Trainner import Trainner
from Predictor import Models
from tqdm import tqdm
import os
import ipdb


def train(**kwargs):
    args = Config()
    args.parse(kwargs)
    loss_func = masked_cross_entropy2
    score_func = batch_scorer
    train_set = DataSet(args.sog_processed+'train/')
    dev_set = DataSet(args.sog_processed+'dev/')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn,num_workers=20, drop_last=True)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
    vocab = pk.load(open('Predictor/Utils/sogou_vocab.pkl', 'rb'))
    eos_id, sos_id = vocab.token2id['<EOS>'], vocab.token2id['<BOS>']
    args.eos_id = eos_id
    args.sos_id = sos_id
    model = getattr(Models, args.model_name)(matrix=vocab.matrix, args=args)
    trainner = Trainner(args, vocab)
    trainner.train(model, loss_func, score_func, train_loader, dev_loader, resume=args.resume)


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


from Predictor import Summarizer
from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('base.html')

@app.route('/', methods=['POST'])
def predict():
    message = request.form['text']
    summ = Summarizer()
    res = summ.predict(message)
    return render_template('processed.html', result=res)


if __name__ == '__main__':
    app.run(debug=True)
