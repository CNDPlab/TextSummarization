import torch as t
from Predictor.Models import Encoder, Decoder



class EncoderDecoder(t.nn.Module):
    def __init__(self, matrix, args):
        super(EncoderDecoder, self).__init__()
        self.vocab_size = matrix.shape[0]
        self.embedding_size = matrix.shape[1]

        self.padding_idx = args.padding_idx
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.sos_id = args.sos_id
        self.eos_id = args.eos_id
        self.decoding_max_lenth = args.decoding_max_lenth
        self.embedding = t.nn.Embedding(self.vocab_size,
                                        self.embedding_size,
                                        padding_idx=self.padding_idx)

        self.encoder = Encoder(cell_type='GRU',
                               input_size=self.embedding_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout=self.dropout)

        self.decoder = Decoder(input_size=self.embedding_size,
                               hidden_size=self.hidden_size,
                               max_lenth=self.decoding_max_lenth,
                               sos_id=self.sos_id,
                               eos_id=self.eos_id,
                               vocab_size=self.vocab_size,
                               num_layer=self.num_layers)

    def forward(self, inputs, lenths, true_seq, use_teacher_forcing):
        net = self.embedding(inputs)
        hidden_states, final_states = self.encoder(net, lenths)

        output_token_list, output_hidden_state_list, output_seq_lenth = self.decoder(true_seq=true_seq,
                                                                   encoder_hidden_states=hidden_states,
                                                                   decoder_init_state=final_states,
                                                                   embedding=self.embedding,
                                                                   use_teacher_forcing=use_teacher_forcing)

        return output_token_list, output_hidden_state_list, output_seq_lenth


# inputs = t.Tensor([[2, 6, 5, 7, 8, 3, 0, 0], [2, 6, 7, 4, 3, 0, 0, 0], [2, 6, 7, 3, 0, 0, 0, 0]]).long()
# matrix = t.nn.Embedding(20, 128).weight
# lenths = t.Tensor([6, 5, 4])
# use_teacher_forcing = True
# true_seq = t.Tensor([[2, 5, 7, 5, 3], [2, 4, 8, 3, 0], [2,4,4,3,0]]).long()
# args = Config()
# encoder_decoder = EncoderDecoder(matrix, args)
# res = encoder_decoder(inputs, lenths, true_seq)
# print(t.stack(res[0], 1).shape)


