import torch as t
import numpy as np
import ipdb
from collections import OrderedDict


class Transformer(t.nn.Module):
    def __init__(self, args, matrix):
        super(Transformer, self).__init__()
        self.num_head = args.num_head
        self.vocabulary_size = matrix.size()[0]
        self.input_size = matrix.size()[1]
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.num_block = 6
        self.encoder_max_lenth = args.encoder_max_lenth
        self.decode_max_lenth = 30
        self.sos_id = args.sos_id
        self.encoder = Encoder(self.num_head, self.input_size, self.hidden_size, self.dropout, self.num_block, matrix, self.encoder_max_lenth)
        self.decoder = Decoder(self.num_head, self.input_size, self.hidden_size, self.dropout, self.num_block, matrix)

    def forward(self, inputs, if_sample=False):
        batch_size = inputs.size()[0]
        input_mask = inputs.ne(0).data
        device = inputs.device

        encoder_outputs = self.encoder(inputs=inputs)
        tokens = t.Tensor([self.sos_id]*batch_size).unsqueeze(-1).long().to(device) # batch of sos_id [B, x]
        probs = t.zeros((batch_size, 1, self.vocabulary_size))
        for i in range(self.decode_max_lenth):
            output_token, output_probs = self.decoder(decoder_inputs=tokens, encoder_outputs=encoder_outputs, encoder_mask=input_mask)
            if not if_sample:
                tokens = t.cat([tokens, output_token], -1)
                probs = t.cat([probs, output_probs], -2)
            else:
                output_token = t.nn.functional.softmax(output_probs, -1).multinomial(1)
                tokens = t.cat([tokens, output_token], -1)
                probs = t.cat([probs, output_probs], -2)

        return tokens[:, 1:], t.nn.functional.log_softmax(probs[:, 1:, :])


    def beam_forward(self, inputs, beam_size):
        pass


class Decoder(t.nn.Module):
    def __init__(self, num_head, input_size, hidden_size, dropout, num_block, matrix):
        super(Decoder, self).__init__()
        self.embedding = t.nn.Embedding(matrix.size()[0], matrix.size()[1], padding_idx=0, _weight=matrix)
        self.position_embedding = PositionEncoding(args.encoder_max_lenth, matrix.size()[1])
        self.decoder_blocks = t.nn.ModuleList([
            DecoderBlock(num_head, input_size, hidden_size, dropout) for _ in range(num_block)
        ])
        self.projection = t.nn.Linear(matrix.size()[1], matrix.size()[0])
        self.projection.weight.data = self.embedding.weight.data

    def get_self_attention_mask(self, inputs):
        device = inputs.device
        batch_size, seqlenth = inputs.size()
        mask = np.tril(np.ones((batch_size, seqlenth, seqlenth)), k=0).astype('uint8')
        mask = t.from_numpy(mask).to(device)
        return mask

    def get_dot_attention_mask(self, inputs, encoder_mask):
        # encoder_mask B
        inputs_mask = inputs.ne(0).data
        dot_attention_mask = t.bmm(inputs_mask.unsqueeze(-1), encoder_mask.data.unsqueeze(-2))
        return dot_attention_mask

    def forward(self, decoder_inputs, encoder_outputs, encoder_mask):
        self_attention_mask = self.get_self_attention_mask(decoder_inputs)
        dot_attention_mask = self.get_dot_attention_mask(decoder_inputs, encoder_mask)
        #inputs B, seqdecode, embedding_size
        #encoder_outputs B, seq encoder , embedding_size
        decoder_inputs_word = self.embedding(decoder_inputs)
        decoder_inputs_posi = self.position_embedding(decoder_inputs)
        decoder_inputs = decoder_inputs_word + decoder_inputs_posi

        for decoder_block in self.decoder_blocks:
            decoder_inputs = decoder_block(decoder_inputs, encoder_outputs, self_attention_mask, dot_attention_mask)

        probs = self.projection(decoder_inputs)[:, -1, :].unsqueeze(-2)
        tokens = probs.topk(1)[1][:, -1]
        return tokens, probs


class DecoderBlock(t.nn.Module):
    def __init__(self, num_head, input_size, hidden_size, dropout):
        super(DecoderBlock, self).__init__()
        self.self_multi_head_attention = MultiHeadAttentionBlock(num_head, input_size, hidden_size, dropout)
        self.dot_multi_head_attention = MultiHeadAttentionBlock(num_head, input_size, hidden_size, dropout)
        self.feed_forward = FeedForward(input_size, hidden_size, dropout)

    def forward(self, inputs, encoder_outputs, attention_mask, dot_attention_mask):
        ipdb.set_trace()
        net = self.self_multi_head_attention(inputs, inputs, inputs, attention_mask)
        net = self.dot_multi_head_attention(net, encoder_outputs, encoder_outputs, dot_attention_mask)# key,value = encoder_output
        net = self.feed_forward(net)
        return net


class Encoder(t.nn.Module):
    def __init__(self, num_head, input_size, hidden_size, dropout, num_block, matrix, encoder_max_lenth):
        super(Encoder, self).__init__()
        self.embedding = t.nn.Embedding(matrix.size()[0], matrix.size()[1], padding_idx=0, _weight=matrix)
        self.position_embedding = PositionEncoding(encoder_max_lenth, matrix.size()[1])
        self.encoder_blocks = t.nn.ModuleList([EncoderBlock(num_head, input_size, hidden_size, dropout)
                                               for _ in range(num_block)])

    def get_attention_mask(self, inputs):
        mask = inputs.ne(0).data
        attention_mask = t.bmm(mask.unsqueeze(-1), mask.unsqueeze(-2))
        return attention_mask

    def forward(self, inputs):
        attention_mask = self.get_attention_mask(inputs)
        word = self.embedding(inputs)
        posi = self.position_embedding(inputs)
        net = word+posi
        for encoder_block in self.encoder_blocks:
            net = encoder_block(net, attention_mask)
        return net

class EncoderBlock(t.nn.Module):
    def __init__(self, num_head, input_size, hidden_size, dropout):
        super(EncoderBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttentionBlock(num_head, input_size, hidden_size, dropout)
        self.feed_forward_net = FeedForward(input_size, hidden_size, dropout)

    def forward(self, inputs, attention_mask):
        attention_net = self.multi_head_attention(inputs, inputs, inputs, attention_mask)
        feed_forward_net = self.feed_forward_net(attention_net)
        return feed_forward_net


class MultiHeadAttentionBlock(t.nn.Module):
    def __init__(self, num_head, input_size, hidden_size, dropout):
        super(MultiHeadAttentionBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(input_size, hidden_size, dropout, num_head)
        self.projection = t.nn.Sequential(OrderedDict([
            ('linear', t.nn.Linear(num_head*hidden_size, input_size)),
            ('drop', t.nn.Dropout(dropout))
        ]))
        self.layer_normalization = LayerNormalization(input_size)

    def forward(self, query, key, value, attention_mask):
        net = self.multi_head_attention(query, key, value, attention_mask)
        net = self.projection(net)
        net += query
        net = self.layer_normalization(net)
        return net


class MultiHeadAttention(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_head):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.drop = t.nn.Dropout(dropout)
        self.key = t.nn.Linear(input_size, hidden_size, bias=False)
        self.query = t.nn.Linear(input_size, hidden_size, bias=False)
        self.value = t.nn.Linear(input_size, hidden_size, bias=False)
        self.self_attention = SelfAttention(hidden_size, dropout)
        t.nn.init.xavier_normal_(self.key.weight)
        t.nn.init.xavier_normal_(self.query.weight)
        t.nn.init.xavier_normal_(self.value.weight)

    def forward(self, query, key, value, attention_mask):
        # B, seqlenth, H
        batch_size = key.size()[0]
        key_ = self.drop(self.key(key))
        query_ = self.drop(self.query(query))
        value_ = self.drop(self.value(value))
        key_ = key_.repeat(self.num_head, 1, 1) # B*num_head, seqlenth, H
        query_ = query_.repeat(self.num_head, 1, 1)
        value_ = value_.repeat(self.num_head, 1, 1)
        attention_mask = attention_mask.repeat(self.num_head, 1, 1)
        output, attention_matrix = self.self_attention(query_, key_, value_, attention_mask)
        # B*num_head, seqlenth, H
        output = t.cat(output.split(batch_size, 0), -1)
        # B seqlenth, H*num_head
        return output

class SelfAttention(t.nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SelfAttention, self).__init__()
        self.C = hidden_size ** 0.5
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, query, key, value, attention_mask):
        # B, seqlenth, H
        attention = t.bmm(query, key.transpose(1, 2)) / self.C
        attention.data.masked_fill_(1 - attention_mask, -float('inf'))
        attention = t.nn.functional.softmax(attention, -1)
        attention.masked_fill_(t.isnan(attention), 0)
        attention = self.dropout(attention)
        output = t.bmm(attention, value)
        return output, attention


class FeedForward(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(FeedForward, self).__init__()
        self.linear = t.nn.Sequential(OrderedDict([
            ('linear1', t.nn.Linear(input_size, hidden_size, bias=False)),
            ('relu', t.nn.ReLU(True)),
            ('drop', t.nn.Dropout(dropout)),
            ('linear2', t.nn.Linear(hidden_size, input_size, bias=False))
        ]))
        self.layer_normalization = LayerNormalization(input_size)

    def forward(self, inputs):
        net = self.linear(inputs)
        net += inputs
        net = self.layer_normalization(net)
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
        result = self.gamma * (inputs - mean) / (std + self.eps) + self.beta
        return result


class PositionEncoding(t.nn.Module):
    def __init__(self, max_lenth, hidden_size):
        super(PositionEncoding, self).__init__()
        self.max_lenth = max_lenth
        self.hidden_size = hidden_size
        self.position_encoding = t.nn.Embedding(max_lenth, hidden_size, padding_idx=0)
        self.init()

    def init(self):
        position_enc = np.array([[pos / np.power(10000, 2 * (j // 2)/self.hidden_size) for j in range(self.hidden_size)] if pos != 0
                                 else np.zeros(self.hidden_size) for pos in range(self.max_lenth+1)])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        self.position_encoding.weight.data = t.from_numpy(position_enc).float()
        self.position_encoding.weight.requires_grad = False

    def token2position(self, inputs):

        device = inputs.device
        batch_size, seq_lenth = inputs.size()
        input_mask = inputs.ne(0).data.long()
        positions = t.range(1, seq_lenth).repeat(batch_size).view(batch_size, seq_lenth).long()
        positions *= input_mask
        positions.to(device)
        return positions

    def forward(self, inputs):
        # inputs [B, max_lenth]
        positions = self.token2position(inputs)
        positions_encoded = self.position_encoding(positions)
        return positions_encoded


if __name__ == '__main__':
    from configs import Config
    import pickle as pk
    vocab = pk.load(open('Predictor/Utils/vocab.pkl', 'rb'))
    args = Config()
    args.sos_id = 5
    matrix = vocab.matrix
    inputs = t.Tensor([[5]+[3]*59+[0]*40, [5]+[3]*99]).long()
    transformer = Transformer(args, matrix)
#    output = transformer(inputs)
    output2 = transformer(inputs, True)
