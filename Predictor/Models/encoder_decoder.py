import torch as t
import random
from Predictor.Models import CustomRnn


class EncoderDecoder(t.nn.Module):
    def __init__(self, matrix, args):
        super(EncoderDecoder, self).__init__()
        self.vocab_size = matrix.shape[0]
        self.embedding_size = matrix.shape[1]
        
        self.padding_idx = args.padding_idx
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.bidirectional = args.bidirectional
        
        self.embedding = t.nn.Embedding(self.vocab_size,
                                        self.embedding_size,
                                        padding_idx=self.padding_idx)
        
        self.encoder = Encoder(cell_type='GRU',
                               input_size=self.embedding_size,
                               hidden_size=self.hidden_size,
                               num_layers=2,
                               dropout=self.dropout)
        if self.bidirectional:
            self.hidden_state_projection = t.nn.Linear(2 * self.hidden_size, self.hidden_size)
        else:
            self.hidden_state_projection = t.nn.Linear(self.hidden_size, self.hidden_size)
        self.decoder = Decoder()
        
    def forward(self, inputs, lenths, use_teacher_forcing):
        net = self.embedding(inputs)
        
        hidden_states, final_states = self.encoder(net, lenths)
        if self.bidirectional:
            decode_init_state = t.cat(final_states.split(1, 1), -1).squeeze(-2)
        else:
            decode_init_state = final_states.squeeze(-2)

        decode_init_state = self.hidden_state_projection(decode_init_state)
        decoder_output_hidden, decoder_output_token = self.decoder(hidden_states, decode_init_state, use_teacher_forcing)
        # TODO : finish
        return net

