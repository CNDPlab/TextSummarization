from torch.nn.functional import log_softmax
import torch as t


def lenth2mask(lenths, max_lenth):
    """
    :param lenths: [B] tensor
    :param max_lenth:  num
    :return: [B,max_lenth] tensor
    """
    if max_lenth == None:
        max_lenth = lenths.data.max()
    batch_size = lenths.size()[0]

    mask = t.range(0, max_lenth-1).long()
    mask = mask.expand(batch_size, max_lenth)
    mask.to(lenths.device)
    lenths_mask = lenths.unsqueeze(-1).expand_as(mask)
    mask = mask < lenths_mask
    return mask

def masked_cross_entropy(inputs, targets, input_mask):
    """
    :param inputs:  [B, maxlenth, vocabulary_size] float
    :param targets:  [B, maxlenth]
    :param input_mask: [B, max_lenth]
    :return: loss tensor [1]
    """
    vocabulary_size = inputs.size()[-1]
    flat_inputs_log = log_softmax(inputs.view(-1, vocabulary_size), dim=-1)
    flat_targets = targets.view(-1, 1)

    losses = t.gather(flat_inputs_log, dim=1, index=flat_targets.long()).view(*targets.size())
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


