import pyhanlp


def Segmentor(input):
    seg = pyhanlp.HanLP.segment(input)
    res = [i.toString().split('/')[0] for i in seg]
    return res

