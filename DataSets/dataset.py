import numpy as np
from torch.utils.data import Dataset,DataLoader
import os
import json
import itertools


class DataSet(Dataset):
    def __init__(self, path):
        super(DataSet, self).__init__()
        self.files = [path + i for i in os.listdir(path)]


    def __getitem__(self, item):
        line = json.load(open(self.files[item]))
        text_len = len(line['text_id'])
        title_len = len(line['title_id'])
        return np.array(line['text_id']), np.array(line['title_id']), text_len, title_len


    def __len__(self):
        return len(self.files)


def own_collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    text_id,title_id,text_len,title_len = zip(*batch)
    #pad batch
    text_id = list(itertools.zip_longest(*text_id, fillvalue=0))
    text_id = np.asarray(text_id).transpose().tolist()
    title_id = list(itertools.zip_longest(*title_id, fillvalue=0))
    title_id = np.asarray(title_id).transpose().tolist()
    return text_id, title_id, text_len, title_len

# def own_collate_fn(batch):
#     batch.sort(key=lambda x: len(x[0]), reverse=True)
#     text_id, title_id, text_len, title_len = zip(*batch)
#     #pad batch
#     pad_text = []
#     max_textlen = len(text_id[0])
#     for i in range(len(text_id)):
#         temp_label = [0] * max_textlen
#         temp_label[:len(text_id[i])] = text_id[i]
#         pad_text.append(temp_label)
#
#     pad_title = []
#     max_ttlen = max(len(i) for i in title_id)
#     for i in range(len(text_id)):
#         temp_label = [0] * max_ttlen
#         temp_label[:len(text_id[i])] = text_id[i]
#         pad_title.append(temp_label)
#
#     return pad_text, pad_title, text_len, title_len


###########################################################
tset = DataSet('sample_processed/train/')
tload = DataLoader(tset,3, collate_fn=own_collate_fn)
text_id, title_id, text_len, title_len = tload.__iter__().__next__()
