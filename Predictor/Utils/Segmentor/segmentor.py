import os
import pyltp
from pyltp import Segmentor, Postagger


class Seg_only(object):
    def __init__(self, seg_model_path=os.path.dirname(__file__)+'/cws.model'):
        self.segmentor = Segmentor()
        self.segmentor.load(seg_model_path)

    def segment(self, input_str):
        words = list(self.segmentor.segment(input_str))
        return words

class Seg_POS(object):
    def __init__(self, seg_model_path=os.path.dirname(__file__)+'/cws.model', pos_model_path=os.path.dirname(__file__)+'/pos.model'):
        self.segmentor = Segmentor()
        self.segmentor.load(seg_model_path)
        self.Postagger = Postagger()
        self.Postagger.load(pos_model_path)

    def segment(self, input_str):
        words = list(self.segmentor.segment(input_str))
        pos = list(self.Postagger.postag(words))
        return words, pos

