from .Segmentor import Seg_only
import re

pattern = re.compile('(\d+(\.\d+)?)')
seg = Seg_only()
stopwords = [line.strip() for line in open('Predictor/Utils/stopwords.dat.txt', 'r', encoding='utf-8').readlines()]



def clean(input):
    input = strq2b(input)
    input = re.sub(pattern, '#', input)
    input = input.replace(u'\u3000', '')
    input = seg.segment(input)
    #去停用词
    input = [word for word in input if word not in stopwords]
    return input

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

def seg_func(input_line):
    text = input_line['text']
    title = input_line['title']
    input_line['text_seg'] = ['<BOS>'] + clean(text) + ['<EOS>']
    input_line['title_seg'] = ['<BOS>'] + clean(title) + ['<EOS>']
    return input_line

def predict_pipeline(input_str):
    token_id = clean(input_str)
    return token_id

