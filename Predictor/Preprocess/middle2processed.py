from Predictor.Preprocess import single_line_pipe_preprocess
from Predictor.Utils import Vocab,segmentor
from configs import Config
import os
import shutil
import pandas as pd
from collections import Counter


args = Config()
vocab = Vocab()
vocab.load_pretrained(path=args.pretrained_path)


def middle_seg_dump_count():
    df =

def init_vocab():
    pass

def gen_processed():
    pass


class Processor(object):
    def __init__(self,args):
        self.args = args
        self.counter = None
        self.middle_files = os.listdir(args.middle_folder)

    def process_files(self):
        for middle_file in self.middle_files:
            self.clean_seg(middle_file)
        self.gen_vocab()
        for processed_file in os.listdir(self.args.processed_folder):
            self.processed2id(processed_file)



    def clean_seg(self,file):
        path = self.args.middle_folder + file
        df = pd.read_json(path)
        df['text_seg'] = df.text.apply(segmentor)
        df['title_seg'] = df.title.apply(segmentor)
        df.to_json(self.args.processed_folder+file)
        if file == 'train.json':
            self.counter = Counter()
            counters = df['text_seg'].apply(Counter())
            for counter in counters:
                self.counter.update(counter)

    def gen_vocab(self):
        assert self.counter is not None


    def processed2id(self, file):
        path = self.args.processed_folder + file
        df = pd.read_json(path)
        df['text_id'] = None


