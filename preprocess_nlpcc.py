import json
import gc
import shutil
import os
import re
import numpy
import pandas as pd
from cytoolz import concatv
from Predictor.Utils import Vocab
from configs import Config
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle as pk
import ipdb
import time
import gensim


start = time.time()
args = Config()
train_file_name = 'Datas/NLPCC/toutiao4nlpcc/train_with_summ.txt'
eval_file_name = 'Datas/NLPCC/toutiao4nlpcc/train_with_summ.txt'
test_file_name = 'Datas/NLPCC/tasktestdata03/tasktestdata03.txt'



with open(train_file_name) as f_:
    train_raw = f_.readlines()
with open(eval_file_name) as f_:
    eval_raw = f_.readlines()
with open(test_file_name) as f_:
    test_raw = f_.readlines()

# train_raw = train_raw[:100]
# eval_raw = eval_raw[:10]
# test_raw = test_raw[:10]

train_df = pd.DataFrame({'summarization': [json.loads(i)['summarization'] for i in train_raw],
                         'article': [json.loads(i)['article'] for i in train_raw]})

eval_df = pd.DataFrame({'summarization': [json.loads(i)['summarization'] for i in eval_raw],
                         'article': [json.loads(i)['article'] for i in eval_raw]})

test_df = pd.DataFrame({'summarization': [json.loads(i)['summarization'] for i in test_raw],
                         'article': [json.loads(i)['article'] for i in test_raw]})

del train_raw, eval_raw, test_raw
gc.collect()

if os.path.exists(args.nlpcc_middle):
    shutil.rmtree(args.nlpcc_middle)

os.mkdir(args.nlpcc_middle)

def is_ustr(in_str):
    out_str = ''
    for i in range(len(in_str)):
        if is_uchar(in_str[i]):
            out_str = out_str+in_str[i]
        else:
            pass
    return out_str


def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
            return True
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
            return True
    if uchar in ('-', ',', '，', '。', '.', '?', ':', ';'):
            return True
    return False

stopwords = [line.strip() for line in open('Predictor/Utils/stopwords.dat.txt', 'r', encoding='utf-8').readlines()]

def remove(text):
    text = text.replace('<Paragraph>', '')
    text = re.sub(r'\(.*\)', '', text)
    text = re.sub(r'\[.*\]', '', text)
    text = re.sub(r'\{.*\}', '', text)
    text = re.sub(r'\【.*\】', '', text)
    return text

def process_data(data):
    data = data[1]
    #data['article'] = is_ustr(data.article.replace('<Paragraph>', ''))
    data['article'] = is_ustr(remove(data.article))
    data['summarization'] = is_ustr(remove(data.summarization))
    data['article_char'] = ['<BOS>'] + [i for i in data['article'] if i not in stopwords] + ['<EOS>']
    data['summarization_char'] = ['<BOS>'] + [i for i in data['summarization'] if i not in stopwords] + ['<EOS>']
    del data['article'], data['summarization']
    line = {i: data[i] for i in data.keys()}
    return line

def middle_process_save(df, set):
    with open(args.nlpcc_middle+set+'.json', 'w') as writer:
        with ProcessPoolExecutor(10) as executor:
            result = executor.map(process_data, df.iterrows())
        nresult = []
        for i in tqdm(result, desc='append'):
            nresult.append(i)
        for res in tqdm(nresult):
            json.dump(res, writer, ensure_ascii=False)
            writer.write('\n')

middle_process_save(eval_df, 'dev')
middle_process_save(test_df, 'test')
middle_process_save(train_df, 'train')

#######

class CharSentance(object):
    def __init__(self, args):
        self.path = args.nlpcc_middle+'train.json'
        with open(self.path) as reader:
            self.lines = reader.readlines()
        self.text_char = [json.loads(i)['article_char'] for i in self.lines]
        self.summary_char = [json.loads(i)['summarization_char'] for i in self.lines]

    def __iter__(self):
        for i in concatv(self.text_char, self.summary_char):
            yield i


sentance = CharSentance(args)
vocab = Vocab()
for i in tqdm(sentance):
    vocab.add_sentance(i)


# model = gensim.models.FastText(size=args.embedding_dim, min_count=2, workers=16)
# model.build_vocab(sentance)
# print('building vocab')
# model.train(sentance, total_examples=model.corpus_count, epochs=model.iter)
#
# args.nlpcc_vocab = 'Predictor/Utils/nlpcc_vocab.pkl'
#
# vocab.filter_rare_word_build_vocab(2)
# vocab.use_pretrained(model)
# vocab.save(args.nlpcc_vocab)

vocab = pk.load(open(args.saved_vocab, 'rb'))


if os.path.exists(args.nlpcc_processed):
    shutil.rmtree(args.nlpcc_processed)
os.mkdir(args.nlpcc_processed)


def convert_save(set='test'):
    os.mkdir(args.nlpcc_processed+set+'/')
    with open(args.nlpcc_middle+set+'.json') as reader:
        for index, line in tqdm(enumerate(reader), desc=set):
            nline = json.loads(line)
            nline['text_id'] = [vocab.from_token_id(i) for i in nline['article_char']]
            nline['summary_id'] = [vocab.from_token_id(i) for i in nline['summarization_char']]
            with open(args.nlpcc_processed+set+'/' + str(index)+'.json', 'w') as writer:
                json.dump(nline, writer, ensure_ascii=False)


convert_save('dev')
convert_save('test')
convert_save('train')
end = time.time()
print(f'use {end-start}s')
