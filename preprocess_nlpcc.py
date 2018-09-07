import json
import numpy

file_name = 'Datas/NLPCC/toutiao4nlpcc/train_with_summ.txt'

#Note <Paragraph> is the tag of '\n' (paragraph) in 'article'.
total_len = []
with open(file_name) as f_:
    line = f_.readline().strip('\r\n')
    while line:
        data = json.loads(line)
        #print '----------------------'
        #print 'summarization: ', data['summarization']
        #print 'article: ', data['article'].replace('<Paragraph>', '\r\n')
        line = f_.readline().strip('\r\n')
        total_len.append(len(data['summarization']))

total_len = sorted(total_len, reverse=False)