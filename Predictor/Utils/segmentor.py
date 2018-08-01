import pyhanlp
import re


def clean(input):
    return input

def Segmentor(input):
    input = clean(input)
    seg = list(pyhanlp.HanLP.segment(input))
    res = [i.word for i in seg]
    return res

