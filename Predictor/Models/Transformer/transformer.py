import torch as t


def position_encoding(inputs):
    #TODO
    pass


class Encoder(t.nn.Module):
    def __init__(self, args, matrix):
        super(Encoder, self).__init__()
        self.num_encoder_block = None
        self.embedding = t.nn.Embedding(matrix.size()[0], matrix.size()[1], padding_idx=args.padding_index)
        self.position_encoding = t.nn.Embedding()
        self.encoder_block = None
        self.encoder_blocks = t.nn.ModuleList([EncoderBlock() for _ in range(self.num_encoder_block)])

    def forward(inputs):

        pass

class EncoderBlock(t.nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()

        self.multi_head_attention = None
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



