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
        self.decode_max_lenth = 50
        self.sos_id = args.sos_id
        self.encoder = Encoder(self.num_head, self.input_size, self.hidden_size, self.dropout, self.num_block, matrix, self.encoder_max_lenth)
        self.decoder = Decoder(self.num_head, self.input_size, self.hidden_size, self.dropout, self.num_block, matrix, self.encoder_max_lenth)
        self.decoder.embedding.weight = self.encoder.embedding.weight
        self.decoder.projection.weight = self.encoder.embedding.weight


    def forward(self, inputs, targets):
        targets = targets[:, :-1].contiguous()
        input_mask = inputs.eq(0).data
        encoder_outputs = self.encoder(inputs)
        tokens, probs = self.decoder(decoder_inputs=targets, encoder_outputs=encoder_outputs, encoder_mask=input_mask)
        return tokens, probs

    def predict(self, inputs, beam_size):
        #Beam Search
        input_mask = inputs.eq(0).data
        batch_size = inputs.size()[0]
        device = inputs.device
        encoder_outputs = self.encoder(inputs)
        init_token = t.nn.Tensor([self.sos_id]*batch_size).unsqueeze(-1).to(device)


        for step in range(self.decode_max_lenth):
            step_token, step_prob = self.decoder(decoder_inputs=init_token, encoder_outputs=encoder_outputs, encoder_mask=input_mask)


    def beam_step(self):
        pass


    #
    # def forward(self, inputs, decode_lenth=None, if_sample=False):
    #     batch_size = inputs.size()[0]
    #     input_mask = inputs.eq(0).data
    #     device = inputs.device
    #
    #     encoder_outputs = self.encoder(inputs=inputs)
    #     tokens = t.LongTensor([self.sos_id]*batch_size).unsqueeze(-1).to(device) # batch of sos_id [B, x]
    #     probs = t.zeros((batch_size, 1, self.vocabulary_size)).to(device)
    #     if decode_lenth is not None:
    #         step_cnt = decode_lenth-1
    #     else:
    #         step_cnt = self.decode_max_lenth
    #     for i in range(step_cnt):
    #         output_token, output_probs = self.decoder(decoder_inputs=tokens, encoder_outputs=encoder_outputs, encoder_mask=input_mask)
    #         if not if_sample:
    #             tokens = t.cat([tokens, output_token], -1)
    #             probs = t.cat([probs, output_probs], -2)
    #         else:
    #             output_token = t.nn.functional.softmax(output_probs, -1).multinomial(1)
    #             tokens = t.cat([tokens, output_token], -1)
    #             probs = t.cat([probs, output_probs], -2)
    #     return tokens[:, 1:], t.nn.functional.log_softmax(probs[:, 1:, :], -1)
    #
    #
    # def beam_forward(self, inputs, beam_size):
    #     pass


class Decoder(t.nn.Module):
    def __init__(self, num_head, input_size, hidden_size, dropout, num_block, matrix, encoder_max_lenth):
        super(Decoder, self).__init__()
        self.embedding = t.nn.Embedding(matrix.size()[0], matrix.size()[1], padding_idx=0, _weight=matrix)
        self.position_embedding = PositionEncoding(encoder_max_lenth, matrix.size()[1])
        self.decoder_blocks = t.nn.ModuleList([
            DecoderBlock(num_head, input_size, hidden_size, dropout) for _ in range(num_block)
        ])
        self.projection = t.nn.Linear(matrix.size()[1], matrix.size()[0])
        self.projection_scale = input_size ** -0.5

    def get_self_attention_mask(self, inputs):
        device = inputs.device
        batch_size, seqlenth = inputs.size()
        #TODO fix
        mask = np.triu(np.ones((batch_size, seqlenth, seqlenth), dtype=np.uint8), k=1)
        mask = t.from_numpy(mask).to(device)
        return mask

    def get_dot_attention_mask(self, inputs, encoder_mask):
        # encoder_mask B
        inputs_mask = inputs.eq(0).data
        dot_attention_mask = t.bmm(inputs_mask.unsqueeze(-1).float(), encoder_mask.data.unsqueeze(-2).float()).byte()
        return dot_attention_mask

    def forward(self, decoder_inputs, encoder_outputs, encoder_mask):
        self_attention_mask = self.get_self_attention_mask(decoder_inputs)
        dot_attention_mask = self.get_dot_attention_mask(decoder_inputs, encoder_mask)
        #inputs B, seqdecode, embedding_size
        #encoder_outputs B, seq encoder , embedding_size
        decoder_inputs_word = self.embedding(decoder_inputs)
        decoder_inputs_posi = self.position_embedding(decoder_inputs)
        decoder_inputs_vector = decoder_inputs_word + decoder_inputs_posi

        for decoder_block in self.decoder_blocks:
            decoder_inputs_vector = decoder_block(decoder_inputs_vector, encoder_outputs, self_attention_mask, dot_attention_mask)
        probs = (self.projection(decoder_inputs_vector) * self.projection_scale)
        tokens = probs.argmax(-1)
        return tokens, probs


class DecoderBlock(t.nn.Module):
    def __init__(self, num_head, input_size, hidden_size, dropout):
        super(DecoderBlock, self).__init__()
        self.self_multi_head_attention = MultiHeadAttentionBlock(num_head, input_size, hidden_size, dropout)
        self.dot_multi_head_attention = MultiHeadAttentionBlock(num_head, input_size, hidden_size, dropout)
        self.feed_forward = FeedForward(input_size, hidden_size, dropout)

    def forward(self, inputs, encoder_outputs, attention_mask, dot_attention_mask):
        net = self.self_multi_head_attention(inputs, inputs, attention_mask)
        net = self.dot_multi_head_attention(net, encoder_outputs, dot_attention_mask)# key,value = encoder_output
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
        mask = inputs.eq(0).data
        attention_mask = t.bmm(mask.unsqueeze(-1).float(), mask.unsqueeze(-2).float()).byte()
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
        attention_net = self.multi_head_attention(inputs, inputs, attention_mask)
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

    def forward(self, query, key, attention_mask):
        net, attention_matrix = self.multi_head_attention(query, key, attention_mask)
        net = self.projection(net)
        net = net + query
        net = self.layer_normalization(net)
        return net


class MultiHeadAttention(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_head):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.drop = t.nn.Dropout(dropout)
        #TODO check
        self.reshape_key = t.nn.Linear(input_size, hidden_size * num_head, bias=False)
        self.reshape_query = t.nn.Linear(input_size, hidden_size * num_head, bias=False)
        self.self_attention = SelfAttention(hidden_size, dropout)
        t.nn.init.xavier_normal_(self.reshape_key.weight)
        t.nn.init.xavier_normal_(self.reshape_query.weight)

    def forward(self, query, key, attention_mask):
        #TODO check
        # B, seqlenth, H
        batch_size, key_lenth, _ = key.size()
        batch_size, query_lenth, _ = query.size()

        key_ = self.reshape_key(key).view(batch_size, key_lenth, self.num_head, self.hidden_size)
        query_ = self.reshape_query(query).view(batch_size, query_lenth, self.num_head, self.hidden_size)

        key_ = key_.permute(2, 0, 1, 3).contiguous().view(-1, key_lenth, self.hidden_size)
        query_ = query_.permute(2, 0, 1, 3).contiguous().view(-1, query_lenth, self.hidden_size)
        key_ = self.drop(key_)
        query_ = self.drop(query_)

        attention_mask = attention_mask.repeat(self.num_head, 1, 1)
        output, attention_matrix = self.self_attention(query_, key_, attention_mask)
        output = output.view(self.num_head, batch_size, query_lenth, self.hidden_size)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, query_lenth, -1)
        return output, attention_matrix


class SelfAttention(t.nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SelfAttention, self).__init__()
        self.C = hidden_size ** 0.5
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, query, key, attention_mask):
        # B, seqlenth, H
        attention = t.bmm(query, key.transpose(1, 2)) / self.C

        attention = attention.masked_fill(attention_mask, -float('inf'))
        attention = t.nn.functional.softmax(attention, -1)
        attention = attention.masked_fill(t.isnan(attention), 0)
        attention = self.dropout(attention)
        output = t.bmm(attention, key)
        return output, attention


class FeedForward(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = t.nn.Conv1d(input_size, hidden_size, 1)
        self.linear2 = t.nn.Conv1d(hidden_size, input_size, 1)
        self.drop = t.nn.Dropout(dropout)
        self.relu = t.nn.ReLU()
        self.layer_normalization = LayerNormalization(input_size)

    def forward(self, inputs):
        net = self.linear1(inputs.transpose(1, 2))
        net = self.relu(net)
        net = self.linear2(net)
        net = net.transpose(1, 2)
        net = self.drop(net)
        net = net + inputs
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
    def __init__(self, max_lenth, embedding_dim):
        super(PositionEncoding, self).__init__()
        self.max_lenth = max_lenth
        self.embedding_dim = embedding_dim
        self.position_encoding = t.nn.Embedding(max_lenth, embedding_dim, padding_idx=0)
        self.init()

    def init(self):
        position_enc = np.array([[pos / np.power(10000, 2 * (j // 2)/self.embedding_dim) for j in range(self.embedding_dim)] if pos != 0
                                 else np.zeros(self.embedding_dim) for pos in range(self.max_lenth+1)])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        self.position_encoding.weight.data = t.from_numpy(position_enc).float()
        self.position_encoding.weight.requires_grad = False

    def token2position(self, inputs):
        device = inputs.device
        batch_size, seq_lenth = inputs.size()
        input_mask = inputs.data.eq(0).long()
        positions = t.range(1, seq_lenth).repeat(batch_size).view(batch_size, seq_lenth).long().to(device)
        positions = positions * input_mask
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

t