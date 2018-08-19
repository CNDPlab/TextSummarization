# part 1 as train ,part3 with score 3,4,5 as test
import pandas as pd
from sklearn.model_selection import train_test_split
from configs import Config
from Predictor.Utils import Vocab
import gc
from tqdm import tqdm
from Predictor.Utils.seg_func import seg
import json


args = Config()

with open(args.raw_folder+'DATA/PART_I.txt') as f:
    train_raw = f.readlines()
with open(args.raw_folder+'DATA/PART_III.txt') as f:
    test_raw = f.readlines()

train_df = pd.DataFrame({'score': [None for _ in range(len(train_raw[2::8]))], 'summary': [i.strip() for i in train_raw[2::8]], 'text': [i.strip() for i in train_raw[5::8]]})
train_df, dev_df = train_test_split(train_df, test_size=0.0005)
test_df = pd.DataFrame({'score': [int(i.strip()[13]) for i in test_raw[1::9]], 'summary': [i.strip() for i in test_raw[3::9]], 'text': [i.strip() for i in test_raw[6::9]]})
test_df = test_df[test_df.score>=3]

del train_raw, test_raw
gc.collect()


char_vocab = Vocab()
word_vocab = Vocab()


def middle_process_save(df, set):
    with open(args.middle_folder+set+'.json') as writer:
        for index, data in tqdm(df.iterrows()):
            processed_data = process_data(data)

            json.dump(processed_data, writer, ensure_ascii=False)
            writer.write('\n')

            char_vocab.add_sentance(processed_data['text_char'])
            char_vocab.add_sentance(processed_data['summary_char'])
            word_vocab.add_sentance(processed_data['text_word'])
            word_vocab.add_sentance(processed_data['summary_word'])


def process_data(data):
    data['text_char'] = ['<BOS>'] + [i for i in data.text] + ['<EOS>']
    data['text_word'] = ['<BOS>'] + seg.segment(data.text) + ['<EOS>']
    data['summary_char'] = ['<BOS>'] + [i for i in data.summary] + ['<EOS>']
    data['summary_word'] = ['<BOS>'] + seg.segment(data.summary) + ['<EOS>']
    line = {i: data[i] for i in data.keys()}
    return line

def gen_emb_convert2id():
    pass

def convert2id():
    pass






