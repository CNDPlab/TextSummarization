from torch.nn.functional import log_softmax
import torch as t
from Predictor.Utils import lenth2mask
import ipdb



def loss_function(inputs, targets):
    vocabulary_size = inputs.size()[-1]
    inputs = inputs.view(-1, vocabulary_size)
    targets = targets[:, 1:].contiguous()
    targets = targets.view(-1)
    loss = t.nn.functional.cross_entropy(inputs, targets, ignore_index=0,)
    return loss
#
#
# def masked_cross_entropy(inputs, targets):
#     """
#     :param inputs:  [B, imaxlenth, vocabulary_size] float
#     :param targets:  [B, tmaxlenth]
#     :param lenths: [B]
#     :return: loss tensor [1]
#     """
#     targets = targets[:, 1:].contiguous()
#     batch_size, inp_max_lenth, vocabulary_size = inputs.size()
#     device = inputs.device
#
#     flat_inputs_log = inputs.contiguous().view(-1, vocabulary_size)
#     flat_targets = targets.view(-1, 1)
#     losses = t.gather(flat_inputs_log, dim=1, index=flat_targets.long()).view(*targets.size())
#     target_mask = targets.ne(0).data.float().to(device)
#     losses = losses * target_mask
#     # losses [B, seqlenth]
#     losses = - (losses.sum(-1)/target_mask.sum(-1)).sum() / batch_size
#     return losses


if __name__ == '__main__':
    inputs = t.Tensor([[[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]], [[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]]])
    targets = t.Tensor([[1, 1], [0, 0]]).long()
    input_lenth = t.Tensor([2, 1]).long()
    target_lenth = t.Tensor([2, 2]).long()
    print(loss_function(inputs=inputs, targets=targets))

    inputs = t.Tensor([[[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]], [[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]]])
    targets = t.Tensor([])
