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
eval_file_name = 'Datas/NLPCC/toutiao4nlpcc_eval/evaluation_with_ground_truth.txt'
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

def strq2b(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:#全角空格直接转换
            inside_code = 32
        if inside_code == 58380:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):#全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def remove(text):
    text = text.replace('<Paragraph>', '。')
    text = text.replace('！', '。')
    text = text.replace('：', ':')
    #text = re.sub(r'\([^)]*\)', '', text)
    #text = re.sub(r'\{.*\}', '', text)
    #text = re.sub(r'\（.*\）', '', text)
    text = re.sub(r'\([^()]*\)', '', text)

    text = re.sub(r'\（[^（）]*\）', '', text)

    text = re.sub(r'\[[^]]*\]', '', text)

    text = re.sub(r'\{[^{}]*\}', '', text)
    text = re.sub(r'\{[^{}]*\}', '', text)

    text = re.sub(r'\【[^【】]*\】', '', text)

    return text

def process_data(data):
    data = data[1]
    #data['article'] = is_ustr(data.article.replace('<Paragraph>', ''))
    data['article'] = is_ustr(remove(strq2b(data.article)))
    data['summarization'] = is_ustr(remove(data.summarization))
    data['article'] = data['article'][:490]
    if len(data['article']) > 450:
        data['article'] = data['article'][:data['article'].rfind('。')+1]
    data['article_char'] = ['<BOS>'] + [i for i in data['article']] + ['<EOS>']
    data['summarization_char'] = ['<BOS>'] + [i for i in data['summarization']] + ['<EOS>']
    del data['article'], data['summarization']
    line = {i: data[i] for i in data.keys()}
    return line

def middle_process_save(df, set):
    with open(args.nlpcc_middle+set+'.json', 'w') as writer:
        with ProcessPoolExecutor(10) as executor:
            result = executor.map(process_data, df.iterrows())
        nresult = []
        for i in tqdm(result, desc='append'):
            if len(i['article_char']) > 17:
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
        for i in tqdm(concatv(self.text_char, self.summary_char)):
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


"""
for i in range(1000):
    raw = open('NLPCC/toutiao4nlpcc_eval/evaluation_with_ground_truth.txt').readlines()
    middle = json.loads(open('nlpcc_middle/dev.json').readlines()[i])
    processed = json.load(open('nlpcc_processed/dev/'+str(i)+'.json'))
    print(''.join(json.loads(raw[i])['article'])[:600])
    print('------------------')
    print(''.join(middle['article_char']))
    print('------------------')
    print(''.join(processed['article_char']))
    print('=====================================')
    inputs = input('next:')
    ...:

    """