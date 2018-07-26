from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

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
print(len(content))
print(len(contenttitle))

import pandas as pd

ct = [i if i != '' else None for i in content]

ctt = [i if i != '' else None for i in contenttitle]

df2 = pd.DataFrame({'text': ct,'title':ctt})

df2 = df2.dropna().drop_duplicates()
df2.to_json('raw/df.json')

df2.to_json('raw/df.json')