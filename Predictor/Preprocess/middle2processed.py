from .single_line_pipe import single_line_pipe
from Predictor.Utils import Vocab
from configs import Config


args = Config()
vocab = Vocab()
vocab.load_pretrained(path=args.pretrained_path)


def middle_seg_dump_count():
    pass

def init_vocab():
    pass

def gen_processed():
    pass



