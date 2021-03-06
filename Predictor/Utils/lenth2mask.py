import torch as t
import ipdb

def lenth2mask(lenths, max_lenth=None):
    """
    :param lenths: [B] tensor
    :param max_lenth:  num
    :return: [B,max_lenth] tensor
    """
    if max_lenth == None:
        max_lenth = lenths.data.max()
    batch_size = lenths.size()[0]
    device = lenths.device
    mask = t.range(0, max_lenth-1).long().to(device)
    mask = mask.expand(batch_size, max_lenth)
    lenths_mask = lenths.unsqueeze(-1).expand_as(mask)
    try:
        mask = mask < lenths_mask
    except:
        ipdb.set_trace()
    return mask


def beam_lenth2mask(lenths, max_lenth=None):
    """
    :param lenths: [B] tensor
    :param max_lenth:  num
    :return: [B,max_lenth] tensor
    """
    device = lenths.device
    mask = t.range(0, max_lenth-1).long().to(device)
    lenths_mask = lenths.expand_as(mask)
    mask = mask < lenths_mask
    return mask
