from Predictor.Utils import lenth2mask
import torch as t
import ipdb


def Attention(encoder_hidden_states, encoder_lenths, step_hidden_state):
    """

    :param encoder_hidden_states:[B, E-seqlenth, hidden_size]
    :param encoder_lenths: [B]
    :param step_hidden_state: [B, 1, hidden_size]
    :return:
    """
    corelation_vector = t.bmm(encoder_hidden_states, step_hidden_state.transpose(-1, -2))
    # corelation_vector [B, E-seqlenth, 1]
    attention_vector = softmax_mask(corelation_vector, encoder_lenths)
    # attention_vector [B, 1, E-seqlenth]

    context_vector = t.bmm(attention_vector.unsqueeze(-2), encoder_hidden_states)
    # context_vector [B, 1, hidden_size]
    return attention_vector, context_vector


def softmax_mask(corelation_vector, encoder_lenths):
    seqlenth = corelation_vector.size()[-2]
    mask = lenth2mask(encoder_lenths, seqlenth)
    mask = 1e-20 * (mask - 1).float()
    attention_vector = t.nn.functional.softmax(corelation_vector.squeeze(-1) + mask, dim=-1)
    return attention_vector