from bs4 import BeautifulSoup
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Predictor.Utils import Segmentor
import gensim
from configs import Config
from collections import Counter
from Predictor.Utils.vocab import Vocab
import json
from cytoolz import concatv
import gc
import os
import pickle as pk
from pathos.multiprocessing import ProcessingPool as Pool
import shutil
import pyhanlp


## raw to middle/seg
args = Config()
with open('raw/corpus.txt') as reader:
    data = reader.readlines()

ct = []
ctt = []

for i in tqdm(data[:500]):
    line = BeautifulSoup(i, 'lxml')
    if line.content != None:
        ct.append(line.content.text)
    elif line.contenttitle != None:
        ctt.append(line.contenttitle.text)

ct = [i if i != '' else None for i in ct]
ctt = [i if i != '' else None for i in ctt]

filted_datas = [{'text': i[0], 'title': i[1]} for i in zip(ct, ctt) if (i[0] is None) | (i[1] is None)]
datas = [{'text': i[0], 'title': i[1]} for i in zip(ct, ctt) if (i[0] is not None) & (i[1] is not None)]


def seg(text):
    return

def func(line):
    print('here1')
    line['text_id'] = [i.word for i in list(pyhanlp.HanLP.segment(line['text']))]
    print('here2')
    line['title_id'] = [i.word for i in list(pyhanlp.HanLP.segment(line['title']))]
    print('here3')
    return line

from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(3) as exe:
    result = exe.map(func, datas)
    for i in tqdm(result):
        print(i)

