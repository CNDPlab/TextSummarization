from torch.nn.functional import log_softmax
import torch as t
from Predictor.Utils import lenth2mask
import ipdb


def masked_cross_entropy(inputs, targets, lenths, target_lenth):
    """
    :param inputs:  [B, imaxlenth, vocabulary_size] float
    :param targets:  [B, tmaxlenth]
    :param lenths: [B]
    :return: loss tensor [1]
    """
    batch_size, inp_max_lenth, vocabulary_size = inputs.size()
    tar_max_lenth = targets.size()[-1]
    device = inputs.device

    vocabulary_size = inputs.size()[-1]

    if inp_max_lenth >= tar_max_lenth:
        inputs = inputs[:, :tar_max_lenth, :]
    else:
        inputs = t.cat([inputs, t.zeros((batch_size, tar_max_lenth-inp_max_lenth, vocabulary_size)).to(device)], dim=-2)


    flat_inputs_log = inputs.contiguous().view(-1, vocabulary_size)
    flat_targets = targets.view(-1, 1)
    try:
        losses = t.gather(flat_inputs_log, dim=1, index=flat_targets.long()).view(*targets.size())
    except:
        ipdb.set_trace()
        print(flat_inputs_log.size())
        print(flat_targets.size())
        print(inputs.size())
        print(targets.size())
    input_mask = lenth2mask(lenths, tar_max_lenth).data.float()
    target_mask = lenth2mask(target_lenth, tar_max_lenth).data.float()
    #TODO check

    losses = losses * input_mask * target_mask
    losses = -losses.sum()/(target_mask.sum())
    return losses


def test():

    inputs = t.Tensor([[[1,0,0],[0,1,1]],])


if __name__ == '__main__':
    test()




