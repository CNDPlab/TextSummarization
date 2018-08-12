from Trainner import Trainner
from tensorboardX import SummaryWriter
import numpy as np


class Evaluator(Trainner):
    def __init__(self, args):
        super(Evaluator, self).__init__(args)

    def evaluate(self, model, test_loader, loss_func, score_func):
        scores = []
        losses = []
        for data in test_loader:
            loss, score = self._data2loss(model, loss_func, data, score_func)
            scores.append(score)
            losses.append(loss.item())

        rouge_l_f1_score = np.mean(scores)
        loss = np.mean(losses)
        return rouge_l_f1_score, loss