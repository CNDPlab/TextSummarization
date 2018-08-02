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


args = Config()
with open('raw/corpus.txt') as reader:
    data = reader.readlines()

ct = []
ctt = []

for i in tqdm(data):
    line = BeautifulSoup(i, 'lxml')
    if line.content != None:
        ct.append(line.content.text)
    elif line.contenttitle != None:
        ctt.append(line.contenttitle.text)

ct = [i if i != '' else None for i in ct]
ctt = [i if i != '' else None for i in ctt]

filted_datas = [{'text': i[0], 'title': i[1]} for i in zip(ct, ctt) if (i[0] is None) | (i[1] is None)]
datas = [{'text': i[0], 'title': i[1]} for i in zip(ct, ctt) if (i[0] is not None) & (i[1] is not None)]


from Predictor.Utils.seg_func import seg_func

from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(30) as exe:
    result = exe.map(seg_func,datas)

n_datas = []
for i in tqdm(result):
    n_datas.append(i)

train, test = train_test_split(n_datas, test_size=0.1, random_state=1)
test, dev = train_test_split(test, test_size=0.5, random_state=1)
del n_datas,datas,data
gc.collect()


if os.path.exists(args.middle_folder):
    shutil.rmtree(args.middle_folder)
os.mkdir(args.middle_folder)

with open(args.middle_folder+'train.json','w') as writer:
    for i in tqdm(train, desc='writing train'):
        json.dump(i, writer, ensure_ascii=False)
        writer.write('\n')
with open(args.middle_folder+'test.json','w') as writer:
    for i in tqdm(test, desc='writing test'):
        json.dump(i, writer, ensure_ascii=False)
        writer.write('\n')

with open(args.middle_folder+'dev.json','w') as writer:
    for i in tqdm(dev, desc='writing dev'):
        json.dump(i, writer, ensure_ascii=False)
        writer.write('\n')

del train,test,dev
gc.collect()

class Sentance(object):
    def __init__(self):
        self.path = 'middle/train.json'
        with open(self.path) as reader:
            self.lines = reader.readlines()
        self.text_seg = [json.loads(i)['text_seg'] for i in self.lines]
        self.title_seg = [json.loads(i)['title_seg'] for i in self.lines]

    def __iter__(self):
        for i in concatv(self.text_seg,self.title_seg):
            yield i

print('generating w2v')
sentance = Sentance()
counter = Counter()
vocab = Vocab()
for i in tqdm(sentance):
    counter.update(Counter(i))
    vocab.add_sentance(i)

model = gensim.models.Word2Vec(size=args.embedding_dim,min_count=5,workers=16, sg=1)
model.build_vocab(sentance)
print('building vocab')
model.train(sentance, total_examples=model.corpus_count, epochs=model.iter)
print('saving w2v')
model.save(f'pretrained_{args.embedding_dim}d_{len(model.wv.vocab)//1000}k.bin')

print('filting vocab loading pretrained')
vocab.filter_rare_word_build_vocab(5)
vocab.use_pretrained(model)
vocab.save(args.saved_vocab)

vocab = pk.load(open(args.saved_vocab, 'rb'))
if os.path.exists(args.processed_folder):
    shutil.rmtree(args.processed_folder)
os.mkdir(args.processed_folder)

if os.path.exists(args.processed_folder + 'train/'):
    shutil.rmtree(args.processed_folder + 'train/')
os.mkdir(args.processed_folder + 'train/')

if os.path.exists(args.processed_folder + 'test/'):
    shutil.rmtree(args.processed_folder + 'test/')
os.mkdir(args.processed_folder + 'test/')

if os.path.exists(args.processed_folder + 'dev/'):
    shutil.rmtree(args.processed_folder + 'dev/')
os.mkdir(args.processed_folder + 'dev/')

i = 0
with open(args.middle_folder+'train.json') as reader:
    for line in tqdm(reader,desc='saving train'):
        nline = json.loads(line)
        nline['text_id'] = [vocab.from_token_id(i) for i in nline['text_seg']]
        nline['title_id'] = [vocab.from_token_id(i) for i in nline['title_seg']]
        with open(args.processed_folder + 'train/' + str(i) +'train.json','w') as writer:
            json.dump(nline, writer, ensure_ascii=False)
            writer.write('\n')
            i += 1

i = 0
with open(args.middle_folder+'test.json') as reader:
    for line in tqdm(reader,desc='saving test'):
        nline = json.loads(line)
        nline['text_id'] = [vocab.from_token_id(i) for i in nline['text_seg']]
        nline['title_id'] = [vocab.from_token_id(i) for i in nline['title_seg']]
        with open(args.processed_folder + 'test/' + str(i) +'test.json','w') as writer:
            json.dump(nline, writer, ensure_ascii=False)
            writer.write('\n')
            i += 1

i = 0
with open(args.middle_folder+'dev.json') as reader:
    for line in tqdm(reader,desc='saving dev'):
        nline = json.loads(line)
        nline['text_id'] = [vocab.from_token_id(i) for i in nline['text_seg']]
        nline['title_id'] = [vocab.from_token_id(i) for i in nline['title_seg']]
        with open(args.processed_folder + 'dev/' + str(i) +'dev.json','w') as writer:
            json.dump(nline, writer, ensure_ascii=False)
            writer.write('\n')
            i += 1


