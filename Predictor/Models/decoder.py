import torch as t
from Predictor.Models.attention import Attention, beam_Attention
import numpy as np
import random
import ipdb
from collections import OrderedDict


class Decoder(t.nn.Module):
    """
    simple rnn decoder without attention , using teacher forcing
    """
    def __init__(self, input_size, hidden_size, max_lenth, sos_id, eos_id, vocab_size, beam_size, num_layer):
        super(Decoder, self).__init__()
        self.max_lenth = max_lenth
        self.vocab_size = vocab_size
        self.embedding_dim = 256
        self.sos_id, self.eos_id = sos_id, eos_id
        self.beam_size = beam_size
        self.use_teacher_forcing = True
        self.rnn = t.nn.GRU(input_size=input_size,
                            hidden_size=hidden_size,
                            bidirectional=False,
                            dropout=0,
                            batch_first=True,
                            num_layers=num_layer
                            )
        self.teacher_forcing = True
        self.merge_context_output = t.nn.Linear(hidden_size*2, hidden_size)
        self.projection0 = t.nn.Linear(hidden_size*2, self.embedding_dim)
        self.projection = t.nn.Linear(self.embedding_dim, self.vocab_size)
        self.projection_scale = self.embedding_dim ** -0.5

        t.nn.init.xavier_normal_(self.merge_context_output.weight)
        t.nn.init.xavier_normal_(self.projection.weight)
        t.nn.init.orthogonal_(self.rnn.weight_hh_l0)
        t.nn.init.orthogonal_(self.rnn.weight_ih_l0)

    def forward(self, true_seq=None,
                encoder_hidden_states=None,
                decoder_init_state=None,
                embedding=None,
                encoder_lenths=None):
        """
        :param true_seq: [b,s]
        :param encoder_hidden_states: [b,s,h]
        :param decoder_init_state: [b,num_layer*num_direction,h]
        :param embedding:
        :return:
        """
        # set sos id as init input_token , encoders last hidden state as init hidden_state
        device = encoder_hidden_states.device
        batch_size = encoder_hidden_states.size()[0]
        hidden_size = encoder_hidden_states.size()[-1]
        token = t.Tensor([self.sos_id]*batch_size).long().to(device)
        hidden_state = decoder_init_state.transpose(0, 1).contiguous()
        context_vector = t.zeros((batch_size, 1, hidden_size)).to(device)
        output_token_list = []
        output_token_list.append(token)
        output_prob_list = []
        output_attention_list = []
        ended_seq_id = []
        output_seq_lenth = {i: true_seq.size()[-1] for i in range(batch_size)}
        for step in range(true_seq.size()[-1]):
            if self.teacher_forcing:
                token = true_seq[:, step]
            else:
                pass
            token, prob, hidden_state, attention_vector, context_vector = self.forward_step(token,
                                                                                            hidden_state,
                                                                                            embedding,
                                                                                            encoder_hidden_states,
                                                                                            encoder_lenths,
                                                                                            context_vector)
            #intra-temporal attention
            # if step > 0:
            #     ipdb.set_trace()
            #     attention_vector = attention_vector / t.sum(t.stack(output_attention_list),dim=0)

            if step != true_seq.size()[-1]:
                output_token_list.append(token)
                output_prob_list.append(prob)
                output_attention_list.append(attention_vector)
                for i, v in enumerate(token):
                    if (i not in ended_seq_id) & (v.item() == self.eos_id):
                        output_seq_lenth[i] = step
                        ended_seq_id.append(i)
                # if len(ended_seq_id) == batch_size:
                #     break
            else:
                output_token_list.append(token)
                output_prob_list.append(prob)
                output_attention_list.append(attention_vector)
                for i, v in enumerate(token):
                    if i not in ended_seq_id:
                        output_seq_lenth[i] = step
                        ended_seq_id.append(i)

        output_seq_lenth = np.asarray([val for key, val in sorted(output_seq_lenth.items())])
        return t.stack(output_token_list).transpose(0, 1), t.cat(output_prob_list, dim=1), t.from_numpy(output_seq_lenth).to(device), t.stack(output_attention_list, dim=1)

    def forward_step(self, input_token, input_hidden_state, embedding, encoder_hidden_states, encoder_lenths, context_vector):
        """
        :param input_token: [Batch_size]
        :param input_hidden_state: [Batch_size,hidden_size]
        :param embedding:
        :param encoder_hidden_states: [B, seqlenth, hidden_size]
        :param encoder_lenths: [B, ]
        :param context_vector: [B, 1, hidden_size]
        :return:
        """

        input_vector = embedding(input_token.unsqueeze(-1))
        # input_vector [B, 1, hidden_size]
        rnn_input = t.cat([input_vector, context_vector], -1)
        # rnn_input [B, 1, hidden_size*2]
        rnn_input = self.merge_context_output(rnn_input)
        output_state, hidden_state = self.rnn(rnn_input, input_hidden_state)
        # output_state [B, 1, hidden_size]
        attention_vector, context_vector = Attention(encoder_hidden_states, encoder_lenths, output_state)
        # attention_vector [B, seq_lenth, 1]
        # contexT_vector [B, seq_lenth, 1]
        output_state = self.projection(self.projection0(t.cat([output_state, context_vector], -1))) * self.projection_scale
        output_prob = t.nn.functional.log_softmax(output_state, dim=-1)
        # output_state = self.projection(t.cat([output_state, context_vector], -1))
        # output_state = t.matmul(output_state, embedding.weight.data.transpose(0,1))
        # output_prob = t.nn.functional.log_softmax(output_state, dim=-1)
        output_token = output_prob.topk(1)[1]
        return output_token.long().squeeze(), output_prob, hidden_state, attention_vector, context_vector

    def check_all_done(self, seqs):
        for seq in seqs:
            if not seq[-1]:
                return False
        return True

    def beam_search_forward(self, input_token, input_hidden_state, embedding, encoder_hidden_states, encoder_lenths, context_vector):

        input_vector = embedding(input_token.unsqueeze(-1))
        rnn_input = t.cat([input_vector, context_vector], -1)
        rnn_input = self.merge_context_output(rnn_input)

        output_state, hidden_state = self.rnn(rnn_input, input_hidden_state)
        attention_vector, context_vector = beam_Attention(encoder_hidden_states, encoder_lenths, output_state)

        output_state = self.projection(self.projection0(t.cat([output_state, context_vector], -1))) * self.projection_scale
        output_token = t.nn.functional.softmax(output_state, dim=-1)
        output_prob = output_token.topk(self.beam_size)[0]
        output_token = output_token.topk(self.beam_size)[1]
        return output_token, output_prob, hidden_state, attention_vector, context_vector

    def beam_search_step(self, decoder_init_state=None, top_seqs=None, embedding=None, encoder_hidden_states=None, encoder_lenths=None):
        all_seqs = []
        device = encoder_hidden_states.device
        hidden_size = encoder_hidden_states.size()[-1]
        context_vector = t.zeros((1, 1, hidden_size)).to(device)
        for seq in top_seqs:
            seq_score = seq[1]
            seq_id = seq[0]
            if seq_id[-1] == self.eos_id:
                all_seqs.append((seq_id, seq_score, True))
                continue
            # get current step using encoder_context & seq
            input_hidden_state = decoder_init_state.transpose(0, 1)
            for i_id in seq_id:
                _word, _prob, input_hidden_state, attention_vector, context_vector = \
                    self.beam_search_forward(t.Tensor([i_id]).long().to(device), input_hidden_state, embedding, encoder_hidden_states, encoder_lenths, context_vector)
            for i in range(self.beam_size):
                temp = seq_id
                word = _word[0][0][i].item()
                word_prob = _prob[0][0][i].item()
                score = seq_score + word_prob
                temp = temp + [word]
                done = (word == self.eos_id)
                all_seqs.append([temp, score, done])
        all_seqs = sorted(all_seqs, key=lambda seq: seq[1], reverse=True)
        topk_seqs = all_seqs[:self.beam_size]
        all_done = self.check_all_done(topk_seqs)
        return topk_seqs, all_done

    def beam_search(self, decoder_init_state, embedding, encoder_hidden_states, encoder_lenths):
        # START
        top_seqs = [[[self.sos_id], 1.0]]
        # loop

        for _ in range(self.max_lenth):
            top_seqs, all_done = self.beam_search_step(decoder_init_state, top_seqs, embedding, encoder_hidden_states, encoder_lenths)
            if all_done:
                break
        top_seq = sorted(top_seqs, key=lambda seq: seq[1], reverse=True)[0]
        seq = top_seq[0]
        prob = top_seq[1]
        return seq, prob





if __name__ == '__main__':
    true_seq = t.Tensor([[2, 6, 4, 4, 4, 3, 0, 0], [2, 4, 4, 4, 3, 0, 0, 0]]).long()
    encoder_hidden_state = t.randn((2, 8, 7))
    decoder_init_state = t.randn((2, 1, 7))
    embedding = t.nn.Embedding(10, 7, padding_idx=0)
    decoder = Decoder(input_size=7, hidden_size=7, max_lenth=5, sos_id=2, eos_id=3, vocab_size=7,beam_size=3, num_layer=1)
    output_token_list, output_hidden_state_list, output_seq_lenth = decoder(true_seq, encoder_hidden_state, decoder_init_state, embedding)
    print(output_token_list)
    print(output_hidden_state_list)
    print(output_hidden_state_list[0].shape)
    print(output_seq_lenth)

    decoder = Decoder(input_size=7, hidden_size=7, max_lenth=5, sos_id=2, eos_id=3, vocab_size=7,beam_size=3,num_layer=1)
    embedding = t.nn.Embedding(10, 7, padding_idx=0)
    beam = decoder.beam_search(decoder_init_state[:1, :, :], embedding)
    print(beam)
