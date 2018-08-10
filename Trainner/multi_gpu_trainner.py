from Trainner import Trainner


class MTrainner(Trainner):
    def __init__(self, args):
        super(MTrainner, self).__init__(args)



    def _train_epoch(self, model, optimizer, loss_func, score_func, train_loader, dev_loader, teacher_forcing_ratio):
        #TODO use multi-gpu style
        pass
