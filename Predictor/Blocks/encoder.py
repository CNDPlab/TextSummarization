import torch as t


class RnnEncoder(t.nn.Module):
    def __init__(self, matrix, args):
        super(RnnEncoder, self).__init__()
        self.args = args
        self.embedding = t.nn.Embedding(matrix.shape[0],matrix.shape[1])
        self.rnn_cell = t.nn.GRUCell(self.args.embedding_dim, self.args.hidden_size)
        self.get_hidden_init()

    def forward(self, input_id):
        input_vec = self.embedding(input_id)

    def get_hidden_init(self):
        return None




