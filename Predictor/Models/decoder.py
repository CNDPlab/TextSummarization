import torch as t


class Decoder(t.nn.Module):
    """
    simple rnn decoder without attention , using teacher forcing
    """
    def __init__(self, input_size, hidden_size, max_lenth, sos_id, eos_id, vocab_size):
        super(Decoder, self).__init__()
        self.max_lenth = max_lenth
        self.vocab_size = vocab_size
        self.sos_id, self.eos_id = sos_id, eos_id
        self.rnn = t.nn.RNN(input_size=input_size,
                            hidden_size=hidden_size,
                            bidirectional=False,
                            dropout=0,
                            batch_first=True,
                            num_layers=num_layer
                            )
        self.projection = t.nn.Sequential(t.nn.Linear(hidden_size, self.vocab_size))

    def forward(self, true_seq=None,
                encoder_hidden_states=None,
                decoder_init_state=None,
                embedding=None,
                use_teacher_forcing=True,
                use_attention=True):
        """
        :param true_seq: [b,s]
        :param encoder_hidden_states: [b,s,h]
        :param decoder_init_state: [b,num_layer*num_direction,h]
        :param embedding:
        :param use_teacher_forcing:bool
        :return:
        """
        # set sos id as init input_token , encoders last hidden state as init hidden_state
        batch_size = encoder_hidden_states.size()[0]
        input_token = t.Tensor([self.sos_id]*batch_size).long()
        input_hidden_state = decoder_init_state.transpose(0, 1)
        output_token_list = []
        output_hidden_state_list = []
        ended_seq_id = []
        if use_teacher_forcing:
            output_seq_lenth = {i: true_seq.size()[-1] for i in range(batch_size)}
            for step in range(true_seq.size()[-1]):
                input_token = true_seq[:, step]  # input_token: [Batch_size]
                input_token, input_hidden_state = self.forward_step(input_token,
                                                                    input_hidden_state,
                                                                    embedding)
                output_token_list.append(input_token)
                output_hidden_state_list.append(input_hidden_state)
                if len(input_token) != 0:
                    for i, v in enumerate(input_token):
                        if (i not in ended_seq_id) & (v.item() == self.eos_id):
                            output_seq_lenth[i] = step
                            ended_seq_id.append(i)
                if len(ended_seq_id) == batch_size:
                    break

        else:
            output_seq_lenth = {i: self.max_lenth for i in range(batch_size)}
            for step in range(self.max_lenth):
                input_token, input_hidden_state = self.forward_step(input_token,
                                                                    input_hidden_state,
                                                                    embedding)
                output_token_list.append(input_token)
                output_hidden_state_list.append(input_hidden_state)

                if len(input_token) != 0:
                    for i, v in enumerate(input_token):
                        if (i not in ended_seq_id) & (v.item() == self.eos_id):
                            output_seq_lenth[i] = step
                            ended_seq_id.append(i)
                if len(ended_seq_id) == batch_size:
                    break
        return output_token_list, output_hidden_state_list, output_seq_lenth

    def forward_step(self, input_token, input_hidden_state, embedding):
        """
        :param input_token: [Batch_size]
        :param input_hidden_state: [Batch_size,hidden_size]
        :param embedding:
        :return:
        """
        input_vector = embedding(input_token.unsqueeze(-1))
        output_state, hidden_state = self.rnn(input_vector, input_hidden_state)
        output_token = self.projection(output_state)
        output_token = t.nn.functional.log_softmax(output_token, dim=-1).topk(1)[1]
        return output_token.long().squezze(), hidden_state


    def reduce_mul(self,l):
        out = 1.0
        for x in l:
            out *= x
        return out

    def check_all_done(self,seqs):
        for seq in seqs:
            if not seq[-1]:
                return False
        return True

    def beam_search_forward(self, input_token, input_hidden_state, embedding):
        input_vector = embedding(input_token.unsqueeze(-1))
        output_state, hidden_state = self.rnn(input_vector, input_hidden_state)
        output_token = self.projection(output_state)
        output_token = t.nn.functional.log_softmax(output_token, dim=-1)
        output_prob = output_token.topk(self.beam_size)[0]
        output_token = output_token.topk(self.beam_size)[1]
        return output_token, output_prob, hidden_state

    def beam_search_step(self, decoder_init_state=None, top_seqs = None, embedding=None):
        all_seqs = []
        for seq in top_seqs:
            seq_score = self.reduce_mul([_score for _, _score in seq])
            seq_id = [id for id, _ in seq]
            if seq_id[-1] == self.eos_id:
                all_seqs.append((seq_id, seq_score, True))
                continue
            # get current step using encoder_context & seq
            input_hidden_state = decoder_init_state.transpose(0, 1)
            current_step = self.beam_search_forward(t.Tensor(seq_id).long(), input_hidden_state, embedding)
            for word,word_prob,_ in enumerate(current_step):
                score = seq_score * word_prob
                rs_seq = seq + [word]
                done = (word == self.eos_id)
                all_seqs.append((rs_seq, score, done))
        all_seqs = sorted(all_seqs, key=lambda seq: seq[1], reverse=True)
        topk_seqs = all_seqs[:self.beam_size]
        all_done = self.check_all_done(topk_seqs)
        return topk_seqs, all_done

    def beam_search(self,decoder_init_state, embedding):
        # START
        top_seqs = [[(self.sos_id, 1.0)]]
        # loop
        for _ in range(self.max_lenth):
            top_seqs, all_done = self.beam_search_step(decoder_init_state, top_seqs, embedding)
            if all_done:
                break
        return top_seqs

    def get_beam_seq(self, decoder_init_state,embedding):
        top_seqs = self.beam_search(decoder_init_state,embedding)
        word_index = []
        word_prob = []
        for i, seq in enumerate(top_seqs):
            for word in seq[1:]:
                word_index.append(word[0])
                word_prob.append(word[1])
                if word_index == self.eos_id:
                    break
        return word_index, self.reduce_mul(word_prob)


def test():
    true_seq = t.Tensor([[2, 6, 4, 4, 4, 3, 0, 0], [2, 4, 4, 4, 3, 0, 0, 0]]).long()
    encoder_hidden_state = t.randn((2, 8, 7))
    decoder_init_state = t.randn((2, 1, 7))
    embedding = t.nn.Embedding(10, 7, padding_idx=0)
    decoder = Decoder(input_size=7, hidden_size=7, max_lenth=5, sos_id=2, eos_id=3, vocab_size=7)
    output_token_list, output_hidden_state_list, output_seq_lenth = decoder(true_seq, encoder_hidden_state, decoder_init_state, embedding,False)
    print(output_token_list)
    print(output_hidden_state_list)
    print(output_seq_lenth)

    decoder = Decoder(input_size=7, hidden_size=7, max_lenth=5, sos_id=2, eos_id=3, vocab_size=7,beam_size=3)
    embedding = t.nn.Embedding(10, 7, padding_idx=0)
    beam = decoder.get_beam_seq(decoder_init_state,embedding)
    print(beam)






if __name__ == '__main__':
    test()
