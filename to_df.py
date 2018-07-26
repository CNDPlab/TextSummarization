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

df = pd.DataFrame({'text': content,'title':contenttitle})
df.to_json('raw/df.json')