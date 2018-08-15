import torch as t
from Predictor.Models import CustomRnn


class Encoder(t.nn.Module):
    def __init__(self, cell_type, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.rnn = CustomRnn(cell_type=cell_type,
                             input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             dropout=dropout
                             )

    def forward(self, inputs, lenths):
        hidden_states, last_states = self.rnn(inputs, lenths)
        return hidden_states, last_states


if __name__ == '__main__':
    inputs = t.Tensor([[1, 2, 3, 0], [4, 6, 0, 0], [3, 0, 0, 0]]).long()
    lenths = t.Tensor([3, 2, 1])
    embedding = t.nn.Embedding(10, 5, padding_idx=0)
    encoder = Encoder('GRU', 5, 5, 2, 0)

    net = embedding(inputs)
    net = encoder(net, lenths)

#TODO bidirectional
