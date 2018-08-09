import os
import time
from Predictor.Utils import Vocab
from tensorboardX import SummaryWriter
import shutil


class Trainner(object):
    def __init__(self, train_loader, dev_loader, test_loader, args, optimizer, vocab, device):
        self.init_time = time.strftime('%Y%m%d%H%M%S')
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.args = args
        self.opt = optimizer
        self.vocab = vocab
        os.mkdir(args.tensorboard_root+self.init_time+'/')
        self.summary_writer = SummaryWriter(args.tensorboard_root+self.init_time+'/')

    def train(self, model, ):
        for i in range(self.args.epochs):
            self.train_epoch()

    def train_epoch(self):
        for step, data in enumerate(self.train_loader):
            self.train_step(data)
        pass


    def train_step(self, model, data, global_step):
        context, title, context_lenth, title_lenth = data
        pass

    def test(self):
        pass

