from .Segmentor import Seg_only
import re

pattern = re.compile(r'[0-9]')
seg = Seg_only()

def clean(input):
    input = re.sub(pattern, '#', input)
    input = input.replace(u'\u3000', '')
    input = seg.segment(input)
    return input



def seg_func(input_line):
    text = input_line['text']
    title = input_line['title']
    input_line['text_seg'] = ['<BOS>'] + clean(text) + ['<EOS>']
    input_line['title_seg'] = ['<BOS>'] + clean(title) + ['<EOS>']
    return input_line