import torch as t
import numpy as np

#TODO check masks
class UniversalTransformer(t.nn.Module):
    def __init__(self, args, matrix):
        super(UniversalTransformer, self).__init__()
        self.args = args

        self.matrix = matrix
        self.matrix[0] = 0
        self.vocabulary_size, self.embedding_size = matrix.size()
        self.max_position_lenth = args.encoder_max_lenth
        self.max_step_lenth = args.max_step_lenth
        self.encoder = Encoder(args.embedding_size, self.vocabulary_size, self.max_position_lenth, self.max_step_lenth,
                               args.num_head, args.hidden_size, args.dropout)
        self.decoder = Decoder(args.embedding_size, self.vocabulary_size, self.max_position_lenth, self.max_step_lenth,
                               args.num_head, args.hidden_size, args.dropout)
        # share weights
        self.encoder.word_embedding.weight = t.from_numpy(matrix)
        self.decoder.word_embedding.weight = self.encoder.word_embedding.weight
        self.decoder.position_embedding.weight = self.encoder.position_embedding.weight
        self.decoder.step_embedding.weight = self.encoder.step_embedding.weight
        self.decoder.projection.weight = self.encoder.word_embedding.weight

    def forward(self, context, ground_truth):
        encoder_output, encoder_input_mask = self.encoder(context)
        decoder_output = self.decoder(encoder_output, encoder_input_mask, ground_truth)
        return decoder_output

    def greedy_search(self, context):
        pass

    def beam_search(self, context):
        pass


class Encoder(t.nn.Module):
    def __init__(self, embedding_size, vocabulary_size, max_position_lenth, max_step_lenth, num_head, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.num_step = max_step_lenth-1
        self.word_embedding = t.nn.Embedding(vocabulary_size, embedding_size)
        self.position_embedding = t.nn.Embedding(max_position_lenth, embedding_size)
        self.step_embedding = t.nn.Embedding(max_step_lenth, embedding_size)
        self.encoder_block = EncoderBlock(num_head, embedding_size, hidden_size, dropout)

    def get_step_feature(self, input_mask, step):
        return input_mask * step

    def get_position_feature(self, input_mask):
        batch_size, seq_lenth = input_mask.size()
        device = input_mask.device
        input_feature = t.range(1, seq_lenth, dtype=t.long, device=device).repeat(batch_size).view(batch_size, seq_lenth)
        input_feature = input_feature * input_mask
        return input_feature
    # masks
    def get_input_mask(self, inputs):
        input_mask = inputs.data.ne(0).long()
        return input_mask

    def get_self_attention_mask(self, input_mask):
        self_attention_mask = t.bmm(input_mask.unsqueeze(-1), input_mask.unsqueeze(-2)).byte()
        return self_attention_mask

    def get_position_mask(self, input_mask, inputs):
        position_mask = input_mask.unsqueeze(-1).expand_as(inputs)
        return position_mask

    def forward(self, inputs):
        input_mask = self.get_input_mask(inputs)
        self_attention_mask = self.get_self_attention_mask(input_mask)
        position_feature = self.get_position_feature(input_mask)
        position_embedding = self.position_embedding(position_feature)
        net = self.word_embedding(inputs)
        position_mask = self.get_position_mask(input_mask, net)

        for step in range(1, self.num_step + 1):
            net = net + position_embedding
            step_feature = self.get_step_feature(input_mask, step)
            step_embedding = self.step_embedding(step_feature)
            net = net + step_embedding
            net = self.encoder_block(net, self_attention_mask, position_mask)
        return net, input_mask


class Decoder(t.nn.Module):
    def __init__(self, embedding_size, vocabulary_size, max_position_lenth, max_step_lenth, num_head, hidden_size, dropout):
        super(Decoder, self).__init__()
        self.num_step = max_step_lenth-1
        self.word_embedding = t.nn.Embedding(vocabulary_size, embedding_size)
        self.position_embedding = PositionEmbedding(max_position_lenth, embedding_size)
        self.step_embedding = StepEmbedding(max_step_lenth, embedding_size)
        self.decoder_block = DecoderBlock(num_head, embedding_size, hidden_size, dropout)
        self.projection = t.nn.Linear(embedding_size, vocabulary_size)

    def get_input_mask(self, inputs):
        input_mask = inputs.data.ne(0).long()
        return input_mask

    def get_self_attention_forward_mask(self, inputs):
        batch_size, seq_lenth = inputs.size()
        device = inputs.device
        self_attention_forward_mask = t.tril(t.ones(seq_lenth, seq_lenth), device=device)
        return self_attention_forward_mask

    def get_self_attention_mask(self, input_mask):
        self_attention_mask = t.bmm(input_mask.unsqueeze(-1), input_mask.unsqueeze(-2))
        return self_attention_mask

    def get_dot_attention_mask(self, encoder_input_mask, decoder_input_mask):
        dot_attention_mask = t.bmm(encoder_input_mask.unsqueeze(-1), decoder_input_mask.unsqueeze(-2))
        return dot_attention_mask

    def get_step_feature(self, input_mask, step):
        return input_mask * step

    def get_position_feature(self, input_mask):
        batch_size, seq_lenth = input_mask.size()
        device = input_mask.device
        input_feature = t.range(1, seq_lenth, dtype=t.long, device=device).repeat(batch_size).view(batch_size, seq_lenth)
        input_feature = input_feature * input_mask
        return input_feature

    def forward(self, encoder_output, encoder_input_mask, inputs):
        input_mask = self.get_input_mask(inputs)
        position_mask = self.get_position_feature(input_mask)
        self_attention_forward_mask = self.get_self_attention_mask(input_mask)
        self_attention_mask = self.get_self_attention_mask(input_mask)
        self_attention_mask = self_attention_mask * self_attention_forward_mask
        dot_attention_mask = self.get_dot_attention_mask(encoder_input_mask, input_mask)

        net = self.word_embedding(inputs)
        position_feature = self.get_position_feature(input_mask)
        position_embedding = self.position_embedding(position_feature)

        for step in range(1, self.num_step+1):
            net = net + position_embedding
            step_feature = self.get_step_feature(input_mask, step)
            step_embedding = self.step_embedding(step_feature)
            net = net + step_embedding
            net = self.decoder_block(net, self_attention_mask, dot_attention_mask, position_mask)
        net = self.projection(net)
        return net


class EncoderBlock(t.nn.Module):
    def __init__(self, num_head, embedding_size, hidden_size, dropout):
        super(EncoderBlock, self).__init__()
        self.multi_head_self_attetnion = MultiHeadSelfAttention(num_head, embedding_size, hidden_size, dropout)
        self.drop_out = t.nn.Dropout(dropout)
        self.layer_normalization1 = LayerNormalization(embedding_size)
        self.transition_function = TransitionFunction(embedding_size, dropout)
        self.layer_normalization2 = LayerNormalization(embedding_size)

    def forward(self, inputs, self_attention_mask, position_mask):
        residual = inputs
        net = self.multi_head_self_attention(inputs, inputs, inputs, self_attention_mask)
        net = net + residual
        net = net * position_mask
        net = self.drop_out(net)
        net = self.layer_normalization1(net)
        residual1 = net
        net = net * position_mask
        net = self.transition_function(net)
        net = net * position_mask
        net = net + residual1
        net = self.drop_out(net)
        net = self.layer_normalization2(net)
        net = net * position_mask
        return net


class DecoderBlock(t.nn.Module):
    def __init__(self, num_head, embedding_size, hidden_size, dropout):
        super(DecoderBlock, self).__init__()
        self.multi_head_self_attetnion = MultiHeadSelfAttention(num_head, embedding_size, hidden_size, dropout)
        self.multi_head_dot_attention = MultiHeadSelfAttention(num_head, embedding_size, hidden_size, dropout)
        self.drop_out = t.nn.Dropout(dropout)
        self.layer_normalization1 = LayerNormalization(embedding_size)
        self.transition_function = TransitionFunction(embedding_size, dropout)
        self.layer_normalization2 = LayerNormalization(embedding_size)
        self.layer_normalization3 = LayerNormalization(embedding_size)

    def forward(self, inputs, encoder_outputs, self_attention_mask, dot_attention_mask, position_mask):
        residual = inputs
        net = self.multi_head_self_attetnion(inputs, inputs, inputs, self_attention_mask)
        net = net + residual
        net = net * position_mask
        net = self.drop_out(net)
        net = self.layer_normalization1(net)
        net = net * position_mask
        residual2 = net
        dot_attention = self.multi_head_dot_attention(encoder_outputs, net, dot_attention_mask)
        dot_attention = dot_attention * position_mask
        net = dot_attention + residual2
        net = self.drop_out(net)
        net = self.layer_normalization2(net)
        residual3 = net
        net = net * position_mask
        net = self.transition_function(net)
        net = net * position_mask
        net = net + residual3
        return net


class PositionEmbedding(t.nn.Module):
    def __init__(self, embedding_size, max_position):
        super(PositionEmbedding, self).__init__()
        self.max_position = max_position
        self.embedding_size = embedding_size
        self.embedding = t.nn.Embedding(max_position, embedding_size)
        self.init()

    def init(self):
        position_enc = np.array([[pos / np.power(10000, 2 * (j // 2)/self.embedding_size) for j in range(self.embedding_size)] if pos != 0
                                 else np.zeros(self.embedding_size) for pos in range(self.max_position+1)])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        self.embedding.weight.data = t.from_numpy(position_enc).float()
        self.embedding.weight.requires_grad = False

    def forward(self, inputs):
        position_embedding = self.embedding(inputs)
        return position_embedding


class StepEmbedding(t.nn.Module):
    def __init__(self, embedding_size, max_step):
        super(StepEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.max_step = max_step
        self.embedding = t.nn.Embedding(max_step, embedding_size)
        self.init()

    def init(self):
        position_enc = np.array([[pos / np.power(10000, 2 * (j // 2)/self.embedding_size) for j in range(self.embedding_size)] if pos != 0
                                 else np.zeros(self.embedding_size) for pos in range(self.max_step+1)])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        self.embedding.weight.data = t.from_numpy(position_enc).float()
        self.embedding.weight.requires_grad = False

    def forward(self, inputs):
        step_embedding = self.embedding(inputs)
        return step_embedding


class MultiHeadSelfAttention(t.nn.Module):
    def __init__(self, num_head, embedding_size, hidden_size, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.reshape_query = t.nn.Linear(embedding_size, hidden_size * num_head, bias=False)
        self.reshape_key = t.nn.Linear(embedding_size, hidden_size * num_head, bias=False)
        self.reshape_value = t.nn.Linear(embedding_size, hidden_size * num_head, bias=False)
        self.self_attention = SelfAttention(hidden_size, dropout)
        self.dropout = t.nn.Dropout(dropout)
        self.projection = t.nn.Linear(hidden_size * num_head, embedding_size)
        t.nn.init.xavier_uniform_(self.reshape_value.weight)
        t.nn.init.xavier_uniform_(self.reshape_key.weight)
        t.nn.init.xavier_uniform_(self.reshape_value.weight)
        t.nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, query, key, value, attention_mask):
        batch_size, query_seqlenth, _ = query.size()
        batch_size, key_seqlenth, _ = key.size()

        query_ = self.reshape_query(query).view(batch_size, query_seqlenth, self.num_head, self.hidden_size)
        key_ = self.reshape_key(key).view(batch_size, key_seqlenth, self.num_head, self.hidden_size)
        value_ = self.reshape_value(value).view(batch_size, key_seqlenth, self.num_head, self.hidden_size)
        query_ = query_.permute(2, 0, 1, 3).contiguous().view(-1, query_seqlenth, self.hidden_size)
        key_ = key_.permute(2, 0, 1, 3).contiguous().view(-1, key_seqlenth, self.hidden_size)
        value_ = value_.permute(2, 0, 1, 3).contiguous().view(-1, key_seqlenth, self.hidden_size)
        query_ = self.dropout(query_)
        key_ = self.dropout(key_)
        value_ = self.dropout(value_)

        attention_mask = attention_mask.repeat(self.num_head*batch_size, 1, 1)
        output, attention_matrix = self.self_attention(query_, key_, value_, attention_mask)
        output = output.view(self.num_head, batch_size, key_seqlenth, self.hidden_size)
        output = output.permute(1, 2, 0, 3).view(batch_size, key_seqlenth, -1)
        output = self.projection(output)
        return output, attention_matrix


class SelfAttention(t.nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SelfAttention, self).__init__()
        self.C = hidden_size ** 0.5
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        attention = t.bmm(query, key.transpose(1, 2))
        attention = attention.masked_fill(1-mask, -float('inf'))
        attention = t.nn.functional.softmax(attention, -1)
        attention = attention.masked_fill(t.isnan(attention), 0)
        attention = self.dropout(attention)
        output = t.bmm(attention, key)
        return output


class LayerNormalization(t.nn.Module):
    def __init__(self, embedding_size, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = t.nn.Parameter(t.ones(embedding_size))
        self.beta = t.nn.Parameter(t.zeros(embedding_size))
        self.eps = eps

    def forward(self, inputs):
        mean = inputs.mean(-1, keepdim=True)
        std = inputs.std(-1, unbiased=False, keepdim=True)
        result = self.gamma * (inputs - mean) / (std + self.eps) + self.beta
        return result


class TransitionFunction(t.nn.Module):
    def __init__(self, embedding_size, dropout):
        super(TransitionFunction, self).__init__()
        self.linear1 = t.nn.Conv1d(embedding_size, embedding_size*4, 1)
        self.linear2 = t.nn.Conv1d(embedding_size*4, embedding_size, 1)
        self.drop = t.nn.Dropout(dropout)
        self.relu = t.nn.ReLU()
        t.nn.init.xavier_normal_(self.linear1.weight)
        t.nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, inputs):
        net = self.linear1(inputs.transpose(1, 2))
        net = self.relu(net)
        net = self.drop(net)
        net = self.linear2(net)
        net = net.transpose(1, 2)
        net = self.drop(net)
        return net

