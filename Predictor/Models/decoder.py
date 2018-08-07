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
                            batch_first=True
                            )
        self.projection = t.nn.Sequential(t.nn.Linear(hidden_size, self.vocab_size))

    def forward(self, true_seq=None,
                encoder_hidden_states=None,
                decoder_init_state=None,
                embedding=None,
                use_teacher_forcing=True):
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
        output_seq_lenth = {i: self.max_lenth for i in range(batch_size)}
        ended_seq_id = []
        for step in range(self.max_lenth):
            if use_teacher_forcing:
                # the first token in true_seq is sos
                input_token = true_seq[:, step]  # input_token: [Batch_size]
                input_token, input_hidden_state = self.forward_step(input_token,
                                                                    input_hidden_state,
                                                                    embedding)
            else:
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
        return output_token.long().squeeze(), hidden_state


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


if __name__ == '__main__':
    test()

