import pickle as pk
from Predictor.Utils import Vocab
from configs import Config

args = Config()

def single_line_pipe_preprocess(input):

    pass


def single_line_pipe_predict(input):
    vocab = pk.load(open(args.saved_vocab,'rb'))
