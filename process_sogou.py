from bs4 import BeautifulSoup
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import gc
import shutil
import os
import re
import numpy as np
import pandas as pd
from cytoolz import concatv
from Predictor.Utils import Vocab
from configs import Config
from tqdm import tqdm
import pickle as pk
import ipdb
import time
import gensim
from concurrent.futures import ProcessPoolExecutor



args = Config()
if not os.path.exists(args.datas_root):
    os.mkdir(args.datas_root)

if os.path.exists(args.sog_raw+'ct.pk'):
    if os.path.exists(args.sog_raw+'ctt.pk'):
        ct = pk.load(open(args.sog_raw+'ct.pk', 'rb'))
        ctt = pk.load(open(args.sog_raw+'ctt.pk', 'rb'))
else:
    with open(args.sog_raw+'corpus.txt') as reader:
        data = reader.readlines()

    ct = []
    ctt = []

    for i in tqdm(data):
        line = BeautifulSoup(i, 'lxml')
        if line.content is not None:
            ct.append(line.content.text)
        elif line.contenttitle is not None:
            ctt.append(line.contenttitle.text)

    ct = [i if i != '' else None for i in ct]
    ctt = [i if i != '' else None for i in ctt]
    pk.dump(ct, open(args.sog_raw+'ct.pk', 'wb'))
    pk.dump(ctt, open(args.sog_raw+'ctt.pk', 'wb'))



###########################################################################
df = pd.DataFrame({'summarization': [i for i in ctt],
                    'article': [i for i in ct]})
df = df.dropna()
df = df.drop_duplicates()

train_df, test_dev_df = train_test_split(df, test_size=0.015)
test_df, eval_df = train_test_split(test_dev_df, test_size=0.7)

del ct, ctt, df
gc.collect()

if os.path.exists(args.sog_middle):
    shutil.rmtree(args.sog_middle)

os.mkdir(args.sog_middle)

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
    if uchar in ('-', ',', '，', '。', '.', '?', ':', ';', '%'):
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
    data['summarization'] = is_ustr(remove(strq2b(data.summarization)))
    data['article'] = data['article'][:400]
    if len(data['article']) > 350:
        data['article'] = data['article'][:data['article'].rfind('。')+1]
    data['article_char'] = ['<BOS>'] + [i for i in data['article']] + ['<EOS>']
    data['summarization_char'] = ['<BOS>'] + [i for i in data['summarization']] + ['<EOS>']
    #del data['article'], data['summarization']
    line = {i: data[i] for i in data.keys()}
    return line

def middle_process_save(df, set):
    with open(args.sog_middle+set+'.json', 'w') as writer:
        with ProcessPoolExecutor(10) as executor:
            result = executor.map(process_data, df.iterrows())
        nresult = []
        for i in tqdm(result, desc='append'):
            if len(i['article_char']) > 100:
                nresult.append(i)
        for res in tqdm(nresult):
            json.dump(res, writer, ensure_ascii=False)
            writer.write('\n')
    if set == 'train':
        middle_df = pd.DataFrame({'summarization': [i['summarization'] for i in nresult],
                    'article': [i['article'] for i in nresult]})
        return middle_df


middle_process_save(eval_df, 'dev')
middle_process_save(test_df, 'test')
middle_df = middle_process_save(train_df, 'train')

#######

corpus = []

for line in tqdm(middle_df.iterrows(), desc='splitting.'):
    for i in line[1].summarization.split('。'):
        if i != '':
            corpus.append(i + '。')
    for i in line[1].article.split('。'):
        if i != '':
            corpus.append(i + '。')

del train_df, middle_df, eval_df, test_df
gc.collect()

class CharSentance(object):
    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for i in tqdm(self.corpus, desc='itering'):
            yield ['<BOS>'] + list(i) + ['<EOS>']


# class CharSentance(object):
#     def __init__(self, args):
#         self.path = args.sog_middle+'train.json'
#         with open(self.path) as reader:
#             self.lines = reader.readlines()
#         self.text_char = [json.loads(i)['article_char'] for i in self.lines]
#         self.summary_char = [json.loads(i)['summarization_char'] for i in self.lines]
#
#     def __iter__(self):
#         for i in tqdm(concatv(self.text_char, self.summary_char)):
#             yield i


sentance = CharSentance(corpus)
del corpus
gc.collect()

vocab = Vocab()
for i in sentance:
    vocab.add_sentance(i)

model = gensim.models.FastText(size=args.embedding_dim, min_count=200, workers=16)
model.build_vocab(sentance)
print('building vocab')
model.train(sentance, total_examples=model.corpus_count, epochs=model.iter)

args.sogou_vocab = 'Predictor/Utils/sogou_vocab.pkl'

vocab.filter_rare_word_build_vocab(200)
vocab.use_pretrained(model)
vocab.save(args.sogou_vocab)

vocab = pk.load(open(args.sogou_vocab, 'rb'))


if os.path.exists(args.sog_processed):
    shutil.rmtree(args.sog_processed)
os.mkdir(args.sog_processed)


def convert_save(set='test'):
    os.mkdir(args.sog_processed+set+'/')
    with open(args.sog_middle+set+'.json') as reader:
        for index, line in tqdm(enumerate(reader), desc=set):
            nline = json.loads(line)
            nline['text_id'] = [vocab.from_token_id(i) for i in nline['article_char']]
            nline['summary_id'] = [vocab.from_token_id(i) for i in nline['summarization_char']]
            with open(args.sog_processed+set+'/' + str(index)+'.json', 'w') as writer:
                json.dump(nline, writer, ensure_ascii=False)


convert_save('dev')
convert_save('test')
convert_save('train')
end = time.time()
print(f'use {end-start}s')









#df = df[(df['text'].apply(len) > 50) & (df['text'].apply(len) < 500)]
# datas = [{'text': v['text'], 'title': v['title']} for i, v in df.iterrows()]
# print(f'total {len(datas)} data')
#
# from Predictor.Utils.seg_func import seg_func
# from concurrent.futures import ProcessPoolExecutor
# with ProcessPoolExecutor(30) as exe:
#     result = exe.map(seg_func, datas)
#
# n_datas = []
# for i in tqdm(result):
#     n_datas.append(i)
#
# train, test = train_test_split(n_datas, test_size=0.1, random_state=1)
# test, dev = train_test_split(test, test_size=0.05, random_state=1)
# del n_datas, datas
# gc.collect()
#
#
# if os.path.exists(args.sog_middle):
#     shutil.rmtree(args.sog_middle)
# os.mkdir(args.sog_middle)
#
# with open(args.sog_middle+'train.json', 'w') as writer:
#     for i in tqdm(train, desc='writing train'):
#         json.dump(i, writer, ensure_ascii=False)
#         writer.write('\n')
#
# with open(args.sog_middle+'test.json', 'w') as writer:
#     for i in tqdm(test, desc='writing test'):
#         json.dump(i, writer, ensure_ascii=False)
#         writer.write('\n')
#
# with open(args.sog_middle+'dev.json', 'w') as writer:
#     for i in tqdm(dev, desc='writing dev'):
#         json.dump(i, writer, ensure_ascii=False)
#         writer.write('\n')
#
# del train, test, dev
# gc.collect()
#
# class Sentance(object):
#     def __init__(self,args):
#         self.path = args.sog_middle+'train.json'
#         with open(self.path) as reader:
#             self.lines = reader.readlines()
#         self.text_seg = [json.loads(i)['text_seg'] for i in self.lines]
#         self.title_seg = [json.loads(i)['title_seg'] for i in self.lines]
#
#     def __iter__(self):
#         for i in concatv(self.text_seg, self.title_seg):
#             yield i
#
# print('generating w2v')
# sentance = Sentance(args)
# counter = Counter()
# vocab = Vocab()
# for i in tqdm(sentance):
#     counter.update(Counter(i))
#     vocab.add_sentance(i)
#
# model = gensim.models.Word2Vec(size=args.embedding_dim, min_count=5, workers=16, sg=1)
# model.build_vocab(sentance)
# print('building vocab')
# model.train(sentance, total_examples=model.corpus_count, epochs=model.iter)
# print('saving w2v')
# model.save(f'{args.datas_root}pretrained_{args.embedding_dim}d_{len(model.wv.vocab)//1000}k.bin')
#
# print('filting vocab loading pretrained')
# vocab.filter_rare_word_build_vocab(5)
# vocab.use_pretrained(model)
# vocab.save(args.saved_vocab)
#
# vocab = pk.load(open(args.saved_vocab, 'rb'))
#
# if os.path.exists(args.processed_folder):
#     shutil.rmtree(args.processed_folder)
# os.mkdir(args.processed_folder)
#
# if os.path.exists(args.processed_folder + 'train/'):
#     shutil.rmtree(args.processed_folder + 'train/')
# os.mkdir(args.processed_folder + 'train/')
#
# if os.path.exists(args.processed_folder + 'test/'):
#     shutil.rmtree(args.processed_folder + 'test/')
# os.mkdir(args.processed_folder + 'test/')
#
# if os.path.exists(args.processed_folder + 'dev/'):
#     shutil.rmtree(args.processed_folder + 'dev/')
# os.mkdir(args.processed_folder + 'dev/')
#
# i = 0
# with open(args.middle_folder+'train.json') as reader:
#     for line in tqdm(reader, desc='saving train'):
#         nline = json.loads(line)
#         nline['text_id'] = [vocab.from_token_id(i) for i in nline['text_seg']]
#         nline['title_id'] = [vocab.from_token_id(i) for i in nline['title_seg']]
#         unk_id = vocab.from_token_id('<UNK>')
#         try:
#             num_unk = dict(Counter(nline['text_id']))[unk_id]
#         except:
#             num_unk = 0
#         if (len(nline['text_id']) > 50) & (len(nline['text_id']) < 500) & (num_unk/len(nline['text_id']) < args.unk_ratio):
#             with open(args.processed_folder + 'train/' + str(i) +'train.json', 'w') as writer:
#                 json.dump(nline, writer, ensure_ascii=False)
#                 writer.write('\n')
#                 i += 1
#
# i = 0
# with open(args.middle_folder+'test.json') as reader:
#     for line in tqdm(reader, desc='saving test'):
#         nline = json.loads(line)
#         nline['text_id'] = [vocab.from_token_id(i) for i in nline['text_seg']]
#         nline['title_id'] = [vocab.from_token_id(i) for i in nline['title_seg']]
#         if (len(nline['text_id']) > 50) & (len(nline['text_id']) < 500):
#             with open(args.processed_folder + 'test/' + str(i) +'test.json', 'w') as writer:
#                 json.dump(nline, writer, ensure_ascii=False)
#                 writer.write('\n')
#                 i += 1
#
# i = 0
# with open(args.middle_folder+'dev.json') as reader:
#     for line in tqdm(reader, desc='saving dev'):
#         nline = json.loads(line)
#         nline['text_id'] = [vocab.from_token_id(i) for i in nline['text_seg']]
#         nline['title_id'] = [vocab.from_token_id(i) for i in nline['title_seg']]
#         if (len(nline['text_id']) > 50) & (len(nline['text_id']) < 500):
#             with open(args.processed_folder + 'dev/' + str(i) +'dev.json','w') as writer:
#                 json.dump(nline, writer, ensure_ascii=False)
#                 writer.write('\n')
#                 i += 1
#
# if not os.path.exists('ckpt'):
#     os.mkdir('ckpt/')
#     os.mkdir(args.tensorboard_root)
#     os.mkdir((args.saved_model_root))



