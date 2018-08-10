from Trainner import Trainner


class Evaluator(Trainner):
    def __init__(self, args):
        super(Evaluator, self).__init__(args)

        pass


    def evaluate(self, model, test_loader):
        for i in test_loader:

        return rouge_l_f1_score, loss

    def evaluate_step(self, ):