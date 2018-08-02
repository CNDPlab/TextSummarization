from .Segmentor import Seg_only

seg = Seg_only()

def seg_func(input_line):
    input_line['text_seg']= ['<BOS>'] + seg.segment(input_line['text'].replace(u'\u3000','')) + ['<EOS>']
    input_line['title_seg'] = ['<BOS>'] + seg.segment(input_line['title'].replace(u'\u3000','')) + ['<EOS>']
    return input_line