import torch as t
import time
import os
import random
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
import ipdb


class Trainner(object):
    def __init__(self, args):
        self.args = args
        self.global_step = 0
        self.global_epoch = 0
        self.init_time = time.strftime('%Y%m%d_%H%M%S')
        self.device = args.device

    def train(self, model, loss_func, score_func, train_loader, dev_loader, teacher_forcing_ratio):
        model.to(self.device)
        optimizer = t.optim.Adam(model.parameters())
        #TODO add model resume func
        os.mkdir(self.args.tensorboard_root+self.init_time+'/')
        self.summary_writer = SummaryWriter(self.args.tensorboard_root+self.init_time+'/')
        print(f'summary writer running in:')
        print(f'tensorboard --logdir {self.args.tensorboard_root+self.init_time}')
        for epoch in range(self.args.epochs):
            self._train_epoch(model, optimizer, loss_func, score_func, train_loader, dev_loader, teacher_forcing_ratio)
            self.global_epoch += 1

        #TODO add save_stratgy.!!using save_state_dict rather than save!!
        self.summary_writer.close()
        print(f'DONE')

    def _train_epoch(self, model, optimizer, loss_func, score_func, train_loader, dev_loader, teacher_forcing_ratio):
        for data in tqdm(train_loader, 'step'):
            if random.random() < teacher_forcing_ratio:
                model.use_teacher_forcing = True
            else:
                model.use_teacher_forcing = False
            if self.global_step > 5000:
                model.use_teacher_forcing = False
            self._train_step(model, optimizer, loss_func, data)
            if self.global_step % self.args.eval_every_step == 0:
                self._eval(model, loss_func, score_func, dev_loader)

    def _train_step(self, model, optimizer, loss_func, data):
        optimizer.zero_grad()
        train_loss = self._data2loss(model, loss_func, data)
        train_loss.backward()
        optimizer.step()

        self.summary_writer.add_scalar('train_loss', train_loss.item(), self.global_step)
        self.global_step += 1

    def _data2loss(self, model, loss_func, data, score_func=None):
        context, title, context_lenths, title_lenths = [i.to(self.device) for i in data]
        token_id, prob_vector, token_lenth = model(context, context_lenths, title)
        loss = loss_func(prob_vector, title, token_lenth)
        if score_func is None:
            return loss
        else:
            #TODO check
            score = score_func(token_id, title)
            return loss, score

    def _eval(self, model, loss_func, score_func, dev_loader):
        eval_losses = []
        eval_scores = []
        model.eval()
        with t.no_grad():
            for data in dev_loader:
                eval_loss, eval_score = self._data2loss(model, loss_func, data, score_func)
                eval_losses.append(eval_loss.item())
                eval_scores.append(eval_score)
        eval_scores = np.mean(eval_scores)
        eval_losses = np.mean(eval_losses)
        self.summary_writer.add_scalar('eval_loss', eval_losses, self.global_step)
        self.summary_writer.add_scalar('eval_score', eval_scores, self.global_step)
        #TODO check belows if is correct
        for i, v in model.named_parameters():
            self.summary_writer.add_histogram(i, v, self.global_step)
        model.train()

    def _save(self, model, path):
        t.save(model.state_dict(), path)

    def _load(self, model, path):
        model.load_state_dict(t.load(path))

