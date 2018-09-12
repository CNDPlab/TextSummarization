import torch as t
import numpy as np
import ipdb
from collections import OrderedDict


class Transformer(t.nn.Module):
    def __init__(self, args, matrix):
        super(Transformer, self).__init__()
        self.matrix = matrix
        self.matrix[0] = 0
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


    def greedy_search(self, inputs, decode_lenth=None, if_sample=False):
        batch_size = inputs.size()[0]
        input_mask = inputs.eq(0).data
        device = inputs.device

        encoder_outputs = self.encoder(inputs=inputs)
        tokens = t.LongTensor([self.sos_id]*batch_size).unsqueeze(-1).to(device) # batch of sos_id [B, x]
        probs = t.zeros((batch_size, 1, self.vocabulary_size)).to(device)
        if decode_lenth is not None:
            step_cnt = decode_lenth-1
        else:
            step_cnt = self.decode_max_lenth
        for i in range(step_cnt):
            output_token, output_probs = self.decoder(decoder_inputs=tokens, encoder_outputs=encoder_outputs, encoder_mask=input_mask)
            if not if_sample:
                tokens = t.cat([tokens, output_token[:, -1:]], -1)
                probs = t.cat([probs, output_probs], -2)
            else:
                output_token = t.nn.functional.softmax(output_probs, -1).multinomial(1)
                tokens = t.cat([tokens, output_token], -1)
                probs = t.cat([probs, output_probs], -2)
        return tokens, t.nn.functional.log_softmax(probs[:, 1:, :], -1)



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
        batch_size, seqlenth = inputs.size()
        subsequent_mask = t.triu(
            t.ones((seqlenth, seqlenth), device=inputs.device, dtype=t.uint8), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)
        return subsequent_mask

    def get_dot_attention_mask(self, inputs, encoder_mask):
        # encoder_mask B
        inputs_mask = inputs.data.ne(0)
        dot_attention_mask = (1-t.bmm(inputs_mask.unsqueeze(-1).float(), 1-encoder_mask.data.unsqueeze(-2).float())).byte()
        return dot_attention_mask

    def get_pad_mask(self, inputs):
        mask = inputs.ne(0).data.float()
        return mask


    def forward(self, decoder_inputs, encoder_outputs, encoder_mask):

        self_attention_mask = self.get_self_attention_mask(decoder_inputs)
        dot_attention_mask = self.get_dot_attention_mask(decoder_inputs, encoder_mask)
        #inputs B, seqdecode, embedding_size
        #encoder_outputs B, seq encoder , embedding_size
        decoder_inputs_word = self.embedding(decoder_inputs)
        decoder_inputs_posi = self.position_embedding(decoder_inputs)
        decoder_inputs_vector = decoder_inputs_word + decoder_inputs_posi
        non_pad_mask = self.get_pad_mask(decoder_inputs).unsqueeze(-1).expand_as(decoder_inputs_vector)
        for decoder_block in self.decoder_blocks:
            decoder_inputs_vector = decoder_block(decoder_inputs_vector, encoder_outputs, self_attention_mask, dot_attention_mask, non_pad_mask)
        probs = (self.projection(decoder_inputs_vector) * self.projection_scale)
        tokens = probs.argmax(-1)
        return tokens, probs


class DecoderBlock(t.nn.Module):
    def __init__(self, num_head, input_size, hidden_size, dropout):
        super(DecoderBlock, self).__init__()
        self.self_multi_head_attention = MultiHeadAttentionBlock(num_head, input_size, hidden_size, dropout)
        self.dot_multi_head_attention = MultiHeadAttentionBlock(num_head, input_size, hidden_size, dropout)
        self.feed_forward = FeedForward(input_size, hidden_size, dropout)

    def forward(self, inputs, encoder_outputs, attention_mask, dot_attention_mask, non_pad_mask):
        net = self.self_multi_head_attention(inputs, inputs, inputs, attention_mask)
        net = net * non_pad_mask
        net = self.dot_multi_head_attention(net, encoder_outputs, encoder_outputs, dot_attention_mask)# key,value = encoder_output
        net = net * non_pad_mask
        net = self.feed_forward(net)
        net = net * non_pad_mask
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
        attention_mask = (1-t.bmm(mask.unsqueeze(-1).float(), mask.unsqueeze(-2).float())).byte()
        return attention_mask

    def get_pad_mask(self, inputs):
        mask = inputs.ne(0).data.float()
        return mask

    def forward(self, inputs):

        attention_mask = self.get_attention_mask(inputs)
        word = self.embedding(inputs)
        posi = self.position_embedding(inputs)
        net = word+posi
        non_pad_mask = self.get_pad_mask(inputs).unsqueeze(-1).expand_as(net)
        for encoder_block in self.encoder_blocks:
            net = encoder_block(net, attention_mask, non_pad_mask)
        return net


class EncoderBlock(t.nn.Module):
    def __init__(self, num_head, input_size, hidden_size, dropout):
        super(EncoderBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttentionBlock(num_head, input_size, hidden_size, dropout)
        self.feed_forward_net = FeedForward(input_size, hidden_size, dropout)

    def forward(self, inputs, attention_mask, non_pad_mask):
        attention_net = self.multi_head_attention(inputs, inputs, inputs, attention_mask)
        attention_net = attention_net * non_pad_mask
        feed_forward_net = self.feed_forward_net(attention_net)
        feed_forward_net = feed_forward_net * non_pad_mask
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
        net, attention_matrix = self.multi_head_attention(query, key, value, attention_mask)
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
        self.reshape_value = t.nn.Linear(input_size, hidden_size * num_head, bias=False)

        self.self_attention = SelfAttention(hidden_size, dropout)
        t.nn.init.xavier_normal_(self.reshape_key.weight)
        t.nn.init.xavier_normal_(self.reshape_query.weight)
        t.nn.init.xavier_normal_(self.reshape_value.weight)

    def forward(self, query, key, value, attention_mask):
        #TODO check
        # B, seqlenth, H
        batch_size, key_lenth, _ = key.size()
        batch_size, query_lenth, _ = query.size()

        key_ = self.reshape_key(key).view(batch_size, key_lenth, self.num_head, self.hidden_size)
        query_ = self.reshape_query(query).view(batch_size, query_lenth, self.num_head, self.hidden_size)
        value_ = self.reshape_value(value).view(batch_size, key_lenth, self.num_head, self.hidden_size)

        key_ = key_.permute(2, 0, 1, 3).contiguous().view(-1, key_lenth, self.hidden_size)
        query_ = query_.permute(2, 0, 1, 3).contiguous().view(-1, query_lenth, self.hidden_size)
        value_ = value_.permute(2, 0, 1, 3).contiguous().view(-1, key_lenth, self.hidden_size)
        key_ = self.drop(key_)
        query_ = self.drop(query_)
        value_ = self.drop(value_)

        attention_mask = attention_mask.repeat(self.num_head, 1, 1)
        output, attention_matrix = self.self_attention(query_, key_, value_, attention_mask)
        output = output.view(self.num_head, batch_size, query_lenth, self.hidden_size)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, query_lenth, -1)
        return output, attention_matrix


class SelfAttention(t.nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SelfAttention, self).__init__()
        self.C = hidden_size ** 0.5
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, query, key, value, attention_mask):
        # B, seqlenth, H
        attention = t.bmm(query, key.transpose(1, 2)) / self.C

        attention = attention.masked_fill(attention_mask, -float('inf'))
        attention = t.nn.functional.softmax(attention, -1)
        attention = attention.masked_fill(t.isnan(attention), 0)
        attention = self.dropout(attention)
        output = t.bmm(attention, value)
        return output, attention


class FeedForward(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = t.nn.Conv1d(input_size, input_size * 3, 1)
        self.linear2 = t.nn.Conv1d(input_size * 3, input_size, 1)
        self.drop = t.nn.Dropout(dropout)
        self.relu = t.nn.ReLU()
        self.layer_normalization = LayerNormalization(input_size)
        t.nn.init.xavier_normal_(self.linear1.weight)
        t.nn.init.xavier_normal_(self.linear2.weight)

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
        input_mask = inputs.data.ne(0).long()
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
    args.sos_id = vocab.token2id['<BOS>']
    args.batch_size=1
    print(args.sos_id)
    matrix = vocab.matrix
    transformer = Transformer(args, matrix)
    mm = t.nn.DataParallel(transformer).cuda()
#    output = transformer(inputs)
#    output2 = transformer(inputs)
    mm.load_state_dict(t.load('ckpt/bak/20180905_031423/saved_models/2018_09_09_13_08_05T0.5764793283712836/model'))
    from torch.utils.data import Dataset, DataLoader
    from DataSets import DataSet
    from DataSets import own_collate_fn
    from Predictor.Utils import batch_scorer
    # train_set = DataSet(args.processed_folder+'train/')
    # dev_set = DataSet(args.processed_folder+'dev/')
    # test_set = DataSet(args.processed_folder+'test/')
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
    # dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
    train_set = DataSet(args.nlpcc_processed+'train/')
    dev_set = DataSet(args.nlpcc_processed+'dev/')
    test_set = DataSet(args.nlpcc_processed+'test/')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
    vocab = pk.load(open('Predictor/Utils/vocab.pkl', 'rb'))
    eos_id, sos_id = vocab.token2id['<EOS>'], vocab.token2id['<BOS>']
    mm.eval()
    with t.no_grad():
        for data in dev_loader:
            context, title, context_lenths, title_lenths = [i.to('cuda') for i in data]
            token_id, prob_vector = mm.module.greedy_search(context)
            score = batch_scorer(token_id.tolist(), title.tolist(), args.eos_id)
            context_word = [[vocab.from_id_token(id.item()) for id in sample] for sample in context]
            words = [[vocab.from_id_token(id.item()) for id in sample] for sample in token_id]
            title_words = [[vocab.from_id_token(id.item()) for id in sample] for sample in title]

            for i in zip(context_word, words, title_words):
                a = input('next')
                context = ''.join(i[0])
                pre = ''.join(i[1])
                tru = ''.join(i[2])
                print('context:')
                print(f'{context}')
                print('------------')
                print('pre:')
                print(f'{pre}')
                print('------------')
                print('tru:')
                print(f'{tru}')
                print('===========================================')
