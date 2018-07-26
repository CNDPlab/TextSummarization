import gensim


class Vocab(object):
    def __init__(self, init_tokens, counter):
        self.init_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.matrix = None
        self.counter = counter

    def load_pretrained(self, path):
        model = gensim.models.KeyedVectors.load_word2vec_format(path)
        id = len(self.init_tokens)

    def tokens2ids(self, tokens):
        pass

    def save(self,path):
        pass
