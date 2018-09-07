import json
import gc
import shutil
import os
import numpy
import pandas as pd
from Predictor.Utils import Vocab
from configs import Config
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

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

train_df = pd.DataFrame({'summarization': [json.loads(i)['summarization'] for i in train_raw],
                         'article': [json.loads(i)['article'] for i in train_raw]})

eval_df = pd.DataFrame({'summarization': [json.loads(i)['summarization'] for i in eval_raw],
                         'article': [json.loads(i)['article'] for i in eval_raw]})

test_df = pd.DataFrame({'summarization': [json.loads(i)['summarization'] for i in test_raw],
                         'article': [json.loads(i)['article'] for i in test_raw]})

del train_raw, eval_raw, test_raw
gc.collect()

char_vocab = Vocab()

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

def process_data(data):
    data = data[1]
    data['article'] = is_ustr(data.text)
    data['summarization'] = is_ustr(data.summary)
    data['article_char'] = ['<BOS>'] + [i for i in data.text if i not in stopwords] + ['<EOS>']
    data['summarization_char'] = ['<BOS>'] + [i for i in data.summary if i not in stopwords] + ['<EOS>']
    del data['text'], data['summary']
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