import re


def str2token(input_string):
    input_string = strq2b(input_string)
    input_string = remove(input_string)
    input_string = is_ustr(input_string)
    input_string = input_string[:400]
    if len(input_string) > 350:
        input_string = input_string[:input_string.rfind('。')+1]
    input_token = ['<BOS>'] + [i for i in input_string] + ['<EOS>']
    return input_token

def is_ustr(in_str):
    out_str = ''
    for i in range(len(in_str)):
        if is_uchar(in_str[i]):
            out_str = out_str+in_str[i]
        else:
            pass
    return out_str


def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    if uchar in ('-', ',', '，', '。', '.', '?', ':', ';', '%'):
        return True
    return False

def strq2b(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        if inside_code == 58380:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def remove(text):
    text = text.replace('<Paragraph>', '。')
    text = text.replace('！', '。')
    text = text.replace('：', ':')
    # text = re.sub(r'\([^)]*\)', '', text)
    # text = re.sub(r'\{.*\}', '', text)
    # text = re.sub(r'\（.*\）', '', text)
    text = re.sub(r'\([^()]*\)', '', text)

    text = re.sub(r'\（[^（）]*\）', '', text)

    text = re.sub(r'\[[^]]*\]', '', text)

    text = re.sub(r'\{[^{}]*\}', '', text)
    text = re.sub(r'\{[^{}]*\}', '', text)

    text = re.sub(r'\【[^【】]*\】', '', text)

    return text







