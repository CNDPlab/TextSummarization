import gensim


class Vocab(object):
    def __init__(self, init_tokens):
        self.init_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.matrix = None

    def load_pretrained(self, path):
        model = gensim.models.KeyedVectors.load_word2vec_format(path)
        id = len(self.init_tokens)
        pass


