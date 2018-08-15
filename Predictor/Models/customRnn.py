import torch as t
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import ipdb


class CustomRnn(t.nn.Module):
    def __init__(self, cell_type, input_size, hidden_size, num_layers, dropout, batch_first=True, bidirectional=False):
        super(CustomRnn, self).__init__()
        assert cell_type in ['GRU', 'RNN', 'LSTM']
        if (num_layers == 1) & (dropout > 0):
            raise ValueError('no drop if 1 layer.')
        self.cell_type = cell_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.rnn = getattr(t.nn, self.cell_type)(input_size=input_size,
                                                 hidden_size=hidden_size,
                                                 num_layers=num_layers,
                                                 dropout=dropout,
                                                 batch_first=batch_first,
                                                 bidirectional=bidirectional)
        t.nn.init.xavier_normal_(self.rnn.bias_hh_l0)
        t.nn.init.xavier_normal_(self.rnn.bias_ih_l0)
        t.nn.init.orthogonal_(self.rnn.weight_hh_l0)
        t.nn.init.orthogonal_(self.rnn.weight_ih_l0)


    def init_hidden_states(self):
        # TODO: add init hidden states with uniform
        pass

    def forward(self, inputs, lenths):
        if isinstance(lenths, t.Tensor):
            lenths = lenths.cpu().tolist()
        device = inputs.device
        sorted_index = np.argsort(-np.array(lenths))
        sorted_lenths = np.array(lenths)[sorted_index]
        unsorted_index = np.argsort(sorted_index)

        sorted_index = t.Tensor(sorted_index).long().to(device)
        sorted_lenths = t.Tensor(sorted_lenths).long().to(device)
        unsorted_index = t.Tensor(unsorted_index).long().to(device)

        sorted_sequences = inputs.index_select(index=sorted_index, dim=0)

        packed = pack_padded_sequence(sorted_sequences, sorted_lenths, batch_first=self.batch_first,)
        hidden_states, last_states = self.rnn(packed)
        hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=self.batch_first)

        hidden_states = hidden_states.index_select(index=unsorted_index, dim=0)
        last_states = last_states.transpose(0, 1).index_select(index=unsorted_index, dim=0)
        return hidden_states, last_states




"""
a = t.Tensor([[1, 3, 0], [1, 2, 2]]).long().cuda()
emb = t.nn.Embedding(10, 5, padding_idx=0).cuda()
a = emb(a)
b = CustomRnn('RNN', input_size=5, hidden_size=5, num_layers=1, dropout=0).cuda()
hidden_states, final_states = b(a, t.Tensor([2, 3]))
print(hidden_states.shape)
print(final_states.shape)
"""

def test():
    a = t.Tensor([[1, 3, 0], [1, 2, 2]]).long()
    emb = t.nn.Embedding(10, 5, padding_idx=0)
    a = emb(a)
    b = CustomRnn('RNN', input_size=5, hidden_size=5, num_layers=1, dropout=0,bidirectional=True)
    hidden_states, final_states = b(a, t.Tensor([2, 3]))
    print(hidden_states.shape)
    print(final_states.shape)

def test_cuda():
    a = t.Tensor([[1, 3, 0], [1, 2, 2]]).long().cuda()
    emb = t.nn.Embedding(10, 5, padding_idx=0).cuda()
    a = emb(a)
    b = CustomRnn('RNN', input_size=5, hidden_size=5, num_layers=1, dropout=0).cuda()
    hidden_states, final_states = b(a, t.Tensor([2, 3]))
    print(hidden_states.shape)
    print(final_states.shape)


if __name__ == '__main__':
    test()