from torch.nn.functional import log_softmax
import torch as t
from Predictor.Utils import lenth2mask
import ipdb


def masked_cross_entropy(inputs, targets, lenths):
    """
    :param inputs:  [B, imaxlenth, vocabulary_size] float
    :param targets:  [B, tmaxlenth]
    :param lenths: [B]
    :return: loss tensor [1]
    """
    #TODO: max lenth of the loss func must
    max_lenth = targets.size()[-1]
    vocabulary_size = inputs.size()[-1]
    inputs = inputs[:, :max_lenth, :]
    flat_inputs_log = log_softmax(inputs.contiguous().view(-1, vocabulary_size), dim=-1)
    flat_targets = targets.view(-1, 1)
    losses = t.gather(flat_inputs_log, dim=1, index=flat_targets.long()).view(*targets.size())
    input_mask = lenth2mask(lenths, max_lenth).data.float()
    losses = losses * input_mask
    losses = -losses.sum()/(input_mask.sum())
    return losses


def test():

    inputs = t.randn((64, 100, 2000))
    targets = t.ones((64, 100))
    input_mask = t.ones((64, 100))
    loss = masked_cross_entropy(inputs, targets, input_mask)
    return loss


if __name__ == '__main__':
    test()


