from collections import Counter
import gensim
import torch as t
import pickle as pk


class Vocab(object):
    def __init__(self, init_token=['<PAD>','<UNK>']):
        self.init_token = init_token
        self.word_counter = {}
        self.offset = len(self.init_token)
        self.token2id = {v:i for i,v in enumerate(self.init_token)}
        self.id2token = {i:v for i,v in enumerate(self.init_token)}

    def add_sentance(self, sentance):
        for word in sentance:
            self.add_word(word)

    def add_word(self, word):
        if word in self.word_counter:
            self.word_counter[word] += 1
        else:
            self.word_counter[word] = 1

    def filter_rare_word_build_vocab(self, min_count):
        common_words = [i for i,v in self.word_counter.items() if v >= min_count]
        print(f'filtered {len(self.word_counter)-len(common_words)} words,{len(common_words)} left')
        for index, word in enumerate(common_words):
            self.token2id[word] = index + self.offset
            self.id2token[index+self.offset] = word

    def from_id_token(self, id):
        return self.id2token[id]

    def from_token_id(self, token):
        try:
            return self.token2id[token]
        except:
            return 1

    def use_pretrained(self, model):
        model = model
        w2v = model.wv
        matrix = t.nn.Embedding(len(self.token2id), model.vector_size).weight
        t.nn.init.xavier_normal_(matrix)

        oovs = []
        with t.no_grad():
            for i in range(len(self.token2id)):
                if self.id2token[i] in w2v:
                    matrix[i, :] = t.Tensor(w2v[self.id2token[i]])
                else:
                    oovs.append(self.id2token[i])
        self.matrix = matrix
        self.oovs = oovs

    def save(self, path):
        pk.dump(self, open(path, 'wb'))