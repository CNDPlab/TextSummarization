import torch as t
import math
#TODO: complete


class PositionalEncoding(t.nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = t.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = t.zeros(max_len, d_model)
        position = t.arange(0, max_len).unsqueeze(1)
        div_term = t.exp(t.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = t.sin(position * div_term)
        pe[:, 1::2] = t.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + t.autograd.Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class Encoder(t.nn.Module):
    def __init__(self, args, matrix):
        super(Encoder, self).__init__()
        self.num_encoder_block = 6
        self.embedding = t.nn.Embedding(matrix.size()[0], matrix.size()[1], padding_idx=args.padding_index)
        self.position_encoding = t.nn.Embedding()
        self.encoder_block = EncoderBlock()
        self.encoder_blocks = t.nn.ModuleList([EncoderBlock() for _ in range(self.num_encoder_block)])

    def forward(self, inputs):
        net = self.embedding(inputs)
        net = self.encoder_blocks(net)
        return net


class EncoderBlock(t.nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()

        self.multi_head_attention = MultiHeadAttentionBlock
        self.feed_forward_net = None
        #TODO layer_normalization

    def forward(self, inputs):
        attention_net = self.multi_head_attention(inputs)
        attention_net += inputs

        feed_forward_net = self.feed_forward_net(attention_net)
        feed_forward_net += attention_net
        return feed_forward_net


class MultiHeadAttentionBlock(t.nn.Module):
    def __init__(self, num_head, hidden_size, input_size):
        super(MultiHeadAttentionBlock, self).__init__()

        self.key = t.nn.Linear(bias=False)
        self.query = t.nn.Linear(bias=False)
        self.value = t.nn.Linear(bias=False)
        self.multi_head_attention = MultiHeadAttention()
        self.layer_normalization = LayerNormalization(hidden_size)

    def forward(self, inputs):
        key = self.key(inputs)
        query = self.query(inputs)
        value = self.value(inputs)

        net = self.multi_head_attention(key, query, value)
        return net


class LayerNormalization(t.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = t.nn.Parameter(t.ones(hidden_size))
        self.beta = t.nn.Parameter(t.zeros(hidden_size))
        self.eps = eps

    def forward(self, inputs):
        mean = inputs.mean(-1, keepdim=True)
        std = inputs.std(-1, unbiased=False, keepdim=True)
        result = self.gama * (inputs - mean) / (std + self.eps) + self.beta
        return result


class MultiHeadAttention(t.nn.Module):
    def __init__(self, hidden_size, drop_ratio):
        super(MultiHeadAttention, self).__init__()
        self.self_attention = SelfAttention(hidden_size, drop_ratio)

    def forward(self, key, query, value, attention_mask):
        # B, seqlenth, H
        pass


class SelfAttention(t.nn.Module):
    def __init__(self, hidden_size, drop_ratio):
        super(SelfAttention, self).__init__()
        self.C = hidden_size ** 0.5
        self.dropout = t.nn.Dropout(drop_ratio)

    def forward(self, key, query, value, attention_mask):
        # B, seqlenth, H
        attention = t.bmm(query, key.transpose(1, 2)) / self.C
        attention.data.masked_fill_(attention_mask, -float('inf'))
        attention = t.nn.functional.softmax(attention, -1)
        attention = self.dropout(attention)
        output = t.bmm(attention, value)
        return output, attention









