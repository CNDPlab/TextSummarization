from Predictor import Summarizer
from Predictor.Utils.data_pipe import str2token
import os
from tqdm import tqdm
import json




with open('Datas/renmrb.txt') as reader:
    datas = reader.readlines()

datas = [json.loads(i) for i in datas]


summ = Summarizer()
outputs = []
for i in tqdm(datas):
    if i['dataContent'] is not None:
        print('==============================================================')
        print(i['dataContent'])
        print('--')
        print(''.join(str2token(i['dataContent'])))
        output = summ.predict(i['dataContent'])
        outputs.append({'context':i['dataContent'], 'predict': output})
        print('---------')
        print(output)
        print('==============================================================')
    else:
        pass

for i in outputs:
    with open('output.txt', 'w') as writer:
        json.dump(i, writer)
        writer.write('\n')
