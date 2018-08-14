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
        self.save_path = args.saved_model_root

    def train(self, model, loss_func, score_func, train_loader, dev_loader, teacher_forcing_ratio, resume):
        print(resume)
        model.to(self.device)
        optimizer = t.optim.Adam(model.parameters())
        if resume:
            loaded = self._load(self.get_latest_cpath(),model)
            optimizer = loaded['optimizer']
            model = loaded['model']
            print(self.global_step,self.global_epoch)
        os.mkdir(self.args.tensorboard_root+self.init_time+'/')
        self.summary_writer = SummaryWriter(self.args.tensorboard_root+self.init_time+'/')
        print(f'summary writer running in:')
        print(f'tensorboard --logdir {self.args.tensorboard_root+self.init_time}')
        for epoch in range(self.args.epochs):
            self._train_epoch(model, optimizer, loss_func, score_func, train_loader, dev_loader, teacher_forcing_ratio)
            self.global_epoch += 1
            self.select_topk_model(3)
        self.summary_writer.close()
        print(f'DONE')

    def _train_epoch(self, model, optimizer, loss_func, score_func, train_loader, dev_loader, teacher_forcing_ratio):
        for data in tqdm(train_loader, desc='train step'):
            if random.random() < teacher_forcing_ratio:
                model.use_teacher_forcing = True
            else:
                model.use_teacher_forcing = False
            if self.global_step > 5000:
                model.use_teacher_forcing = False
            self._train_step(model, optimizer, loss_func, data)
            if self.global_step % self.args.eval_every_step == 0:
                score = self._eval(model, loss_func, score_func, dev_loader)
                if self.global_step % self.args.save_every_step == 0:
                    self._save(model, self.global_epoch, self.global_step, optimizer, score)

    def _train_step(self, model, optimizer, loss_func, data):
        optimizer.zero_grad()
        train_loss = self._data2loss(model, loss_func, data)
        train_loss.backward()
        optimizer.step()

        self.summary_writer.add_scalar('loss/train_loss', train_loss.item(), self.global_step)
        self.global_step += 1

    def _data2loss(self, model, loss_func, data, score_func=None):
        context, title, context_lenths, title_lenths = [i.to(self.device) for i in data]
        token_id, prob_vector, token_lenth, attention_matrix = model(context, context_lenths, title)
        loss = loss_func(prob_vector, title, token_lenth)
        if score_func is None:
            return loss
        else:
            score = score_func(token_id, title)
            return loss, score

    def _eval(self, model, loss_func, score_func, dev_loader):
        eval_losses = []
        eval_scores = []
        model.eval()
        model.use_teacher_forcing = False
        with t.no_grad():
            for data in tqdm(dev_loader, desc='dev_step'):
                eval_loss, eval_score = self._data2loss(model, loss_func, data, score_func)
                eval_losses.append(eval_loss.item())
                eval_scores.append(eval_score)
        eval_scores = np.mean(eval_scores)
        eval_losses = np.mean(eval_losses)
        self.summary_writer.add_scalar('loss/eval_loss', eval_losses, self.global_step)
        self.summary_writer.add_scalar('score/eval_score', eval_scores, self.global_step)
        #TODO check belows if is correct
        for i, v in model.named_parameters():
            self.summary_writer.add_histogram(i.replace('.', '/'), v.clone().cpu().data.numpy(), self.global_step)
        model.train()
        return eval_scores

    def _save(self, model, epoch, step, optimizer,score):
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        info = date_time + '_' + str(score)
        path = os.path.join(self.save_path, info)
        if not os.path.exists(path):
            os.mkdir(path)
        t.save({'epoch': epoch,
                'step': step,
                'optimizer': optimizer
                }, os.path.join(path, 'trainer_state'))
        t.save(model.state_dict(), os.path.join(path, 'model'))

    def _load(self, path, model):
        resume_checkpoint = t.load(os.path.join(path, 'trainer_state'))
        model.load_state_dict(t.load(os.path.join(path, 'model')))
        return {'epoch': resume_checkpoint['epoch'],
                'step': resume_checkpoint['step'],
                'optimizer': resume_checkpoint['optimizer'],
                'model': model}

    def get_latest_cpath(self):
        all_times = sorted(os.listdir(self.save_path), reverse=True)
        return os.path.join(self.save_path, all_times[0])

    def select_topk_model(self, k):
        file_name = os.listdir(self.save_path)
        remove_file = sorted(file_name, key=lambda x: x.split('_')[1], reverse=True)[k:]
        for i in remove_file:
            os.remove(os.path.join(self.save_path, i))
