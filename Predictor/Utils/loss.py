from torch.nn.functional import log_softmax
from configs import Config
import torch as t
from Predictor.Utils import lenth2mask
from Predictor.Utils import batch_scorer
import numpy as np
import ipdb


def masked_cross_entropy(inputs, targets, target_lenth):
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

    flat_inputs_log = inputs.contiguous().view(-1, vocabulary_size)
    flat_targets = targets.contiguous().view(-1, 1)
    losses = t.gather(flat_inputs_log, dim=1, index=flat_targets.long()).view(*targets.size())
    target_mask = lenth2mask(target_lenth, tar_max_lenth).data.float()
    losses = losses * target_mask
    # losses [B, seqlenth]
    losses = - (losses.sum(-1)/target_mask.sum(-1)).sum() / batch_size
    return losses

def mixed_loss(id, inputs, sample_id, sample_inputs, targets, target_lenth):
    """
    :param inputs:  [B, imaxlenth, vocabulary_size] float
    :param targets:  [B, tmaxlenth]
    :param lenths: [B]
    :return: loss tensor [1]
    """
    args = Config()
    batch_size, inp_max_lenth, vocabulary_size = inputs.size()
    targets = targets[:, 1:]
    target_lenth = target_lenth - 1

    tar_max_lenth = targets.size()[-1]
    device = inputs.device
    vocabulary_size = inputs.size()[-1]

    flat_inputs_log = inputs.contiguous().view(-1, vocabulary_size)
    flat_targets = targets.contiguous().view(-1, 1)
    losses = t.gather(flat_inputs_log, dim=1, index=flat_targets.long()).view(*targets.size())
    target_mask = lenth2mask(target_lenth, tar_max_lenth).data.float()
    losses = losses * target_mask
    # losses [B, seqlenth]
    losses = - (losses.sum(-1)/target_mask.sum(-1)).sum() / batch_size

    #sample loss
    flat_sample_log = sample_inputs.contiguous().view(-1, vocabulary_size)
    sample_losses = t.gather(flat_sample_log, dim=1, index=flat_targets.long()).view(*targets.size())

    sample_losses = sample_losses * target_mask
    # losses [B, seqlenth]
    sample_losses = - (sample_losses.sum(-1)/target_mask.sum(-1)).sum() / batch_size
    output_rouge = t.from_numpy(np.array(batch_scorer(id, targets, args.eos_id))).float()
    sample_rouge = t.from_numpy(np.array(batch_scorer(sample_id, targets, args.eos_id))).float()
    rouge = -((output_rouge - sample_rouge)*100).to(device)
    mixed_losses = rouge * sample_losses
    return mixed_losses, losses


if __name__ == '__main__':
    inputs = t.Tensor([[[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]], [[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]]])
    targets = t.Tensor([[1, 1], [0, 0]]).long()
    input_lenth = t.Tensor([2, 1]).long()
    target_lenth = t.Tensor([2, 2]).long()
    print(masked_cross_entropy(inputs=t.log(inputs), targets=targets, lenths=input_lenth, target_lenth=target_lenth))

    inputs = t.Tensor([[[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]], [[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]]])
    targets = t.Tensor([])
