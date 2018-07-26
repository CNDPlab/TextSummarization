from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


with open('raw/corpus.txt') as reader:
    data = reader.readlines()

content = []
contenttitle = []

for i in tqdm(data):
    line = BeautifulSoup(i, 'lxml')
    if line.content != None:
        content.append(line.content.text)
    elif line.contenttitle != None:
        contenttitle.append(line.contenttitle.text)


ct = [i if i != '' else None for i in content]
ctt = [i if i != '' else None for i in contenttitle]
df2 = pd.DataFrame({'text': ct,'title':ctt})
df2 = df2.dropna().drop_duplicates()

train, test = train_test_split(df2, test_size=0.1)
test, dev = train_test_split(test, test_size=0.5)

import os
os.mkdir('middle/')

train.to_json('middle/train.json')
test.to_json('middle/test.json')
dev.to_json('middle/dev.json')