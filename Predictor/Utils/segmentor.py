import pyhanlp
import re


def clean(input):
    return input

def Segmentor(input):
    input = clean(input)
    seg = pyhanlp.HanLP.segment(input)
    res = [i.toString().split('/')[0] for i in seg]
    return res

