import torch as t
from Predictor.Utils.data_pipe import str2token
from Predictor.Utils.vocab import Vocab
import pickle as pk
from Predictor.Models import TransformerREST
import ipdb
from configs import Config

args = Config()


class Summarizer(object):
    def __init__(self):
        self.vocab = pk.load(open('Predictor/Utils/sogou_vocab.pkl', 'rb'))
        self.matrix = self.vocab.matrix
        args.sos_id = self.vocab.token2id['<BOS>']
        args.eos_id = self.vocab.token2id['<EOS>']
        self.model = TransformerREST(args, self.matrix)
        self.model.load_state_dict(t.load('Predictor/cpu_model'))
        self.model.eval()

    def predict(self, input_string):
        input_token = str2token(input_string)
        le = len(input_token)
        input_id = [self.vocab.from_token_id(j) for j in input_token] + [self.vocab.from_token_id('<PAD>') for i in range(400 - le)]
        context = t.Tensor([input_id]).long()
        with t.no_grad():
            predict_token, _ = self.model.beam_search(context)
        words = [[self.vocab.from_id_token(id.item()) for id in sample] for sample in predict_token]
        return ''.join(words[0])

    def predict_lines(self, input_strings):
        input_tokens = [i for i in str2token(input_strings)]
        input_ids = [[self.vocab.from_token_id(j) for j in input_token] + [self.vocab.from_token_id('<PAD>') for i in range(400 - le)] for input_token in input_tokens]
        context = t.Tensor([input_ids]).long()
        with t.no_grad():
            predict_token, _ = self.model.beam_search(context)
        words = [[self.vocab.from_id_token(id.item()) for id in sample] for sample in predict_token]
        return [''.join(i) for i in words]

