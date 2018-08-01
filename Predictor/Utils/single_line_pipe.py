from Predictor.Utils import Segmentor
import pickle as pk
from Predictor.Utils import Vocab
from configs import Config

args = Config()

vocab = pk.load(open(args.saved_vocab, 'rb'))


def single_line_pipe(text_str):
    text_seg = Segmentor(text_str)
    text_id = [vocab.from_token_id(i) for i in text_seg]
    return text_id


