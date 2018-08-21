# part 1 as train ,part3 with score 3,4,5 as test
import pandas as pd
from sklearn.model_selection import train_test_split
from configs import Config
from Predictor.Utils import Vocab
import gc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import json
import shutil
import os
from cytoolz import concatv
import pickle as pk
import gensim




args = Config()

with open(args.raw_folder+'DATA/PART_I.txt') as f:
    train_raw = f.readlines()
with open(args.raw_folder+'DATA/PART_III.txt') as f:
    test_raw = f.readlines()

train_df = pd.DataFrame({'score': [None for _ in range(len(train_raw[2::8]))], 'summary': [i.strip() for i in train_raw[2::8]], 'text': [i.strip() for i in train_raw[5::8]]})
train_df, dev_df = train_test_split(train_df, test_size=0.0005)
test_df = pd.DataFrame({'score': [int(i.strip()[13]) for i in test_raw[1::9]], 'summary': [i.strip() for i in test_raw[3::9]], 'text': [i.strip() for i in test_raw[6::9]]})
test_df = test_df[test_df.score >= 3]

del train_raw, test_raw
gc.collect()


char_vocab = Vocab()
word_vocab = Vocab()

if os.path.exists(args.middle_folder):
    shutil.rmtree(args.middle_folder)

os.mkdir(args.middle_folder)


def middle_process_save(df, set):
    with open(args.middle_folder+set+'.json', 'w') as writer:
        with ProcessPoolExecutor(10) as executor:
            result = executor.map(process_data, df.iterrows())
        nresult = []
        for i in tqdm(result,desc='append'):
            nresult.append(i)
        for res in tqdm(nresult):
            json.dump(res, writer, ensure_ascii=False)
            writer.write('\n')


def process_data(data):
    data = data[1]
    data['text_char'] = ['<BOS>'] + [i for i in data.text] + ['<EOS>']
    data['summary_char'] = ['<BOS>'] + [i for i in data.summary] + ['<EOS>']
    del data['text'], data['summary']
    line = {i: data[i] for i in data.keys()}
    return line


middle_process_save(dev_df, 'dev')
middle_process_save(test_df, 'test')
middle_process_save(train_df, 'train')

##########################################


class CharSentance(object):
    def __init__(self, args):
        self.path = args.middle_folder+'train.json'
        with open(self.path) as reader:
            self.lines = reader.readlines()
        self.text_char = [json.loads(i)['text_char'] for i in self.lines]
        self.summary_char = [json.loads(i)['summary_char'] for i in self.lines]

    def __iter__(self):
        for i in concatv(self.text_char, self.summary_char):
            yield i


sentance = CharSentance(args)
vocab = Vocab()
for i in tqdm(sentance):
    vocab.add_sentance(i)


model = gensim.models.Word2Vec(size=args.embedding_dim, min_count=5, workers=16, sg=1)
model.build_vocab(sentance)
print('building vocab')
model.train(sentance, total_examples=model.corpus_count, epochs=model.iter)
print('saving w2v')

vocab.filter_rare_word_build_vocab(5)
vocab.use_pretrained(model)
vocab.save(args.saved_vocab)

vocab = pk.load(open(args.saved_vocab, 'rb'))


if os.path.exists(args.processed_folder):
    shutil.rmtree(args.processed_folder)
os.mkdir(args.processed_folder)

def convert_save(set='test'):
    os.mkdir(args.processed_folder+set+'/')
    with open(args.middle_folder+set+'.json') as reader:
        for index, line in tqdm(enumerate(reader), desc=set):
            nline = json.loads(line)
            nline['text_id'] = [vocab.from_token_id(i) for i in nline['text_char']]
            nline['summary_id'] = [vocab.from_token_id(i) for i in nline['summary_char']]
            with open(args.processed_folder+set+'/' + str(index)+'.json', 'w') as writer:
                json.dump(nline, writer, ensure_ascii=False)

convert_save('dev')
convert_save('test')
convert_save('train')