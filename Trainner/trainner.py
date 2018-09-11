import torch as t
import time
import os
import random
from tensorboardX import SummaryWriter
from Predictor.Utils.loss import masked_cross_entropy, mixed_loss, masked_cross_entropy2
import numpy as np
from tqdm import tqdm
import shutil
import ipdb


class ScheduledOptim(object):
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, n_current_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = n_current_steps
        self.init_lr = np.power(d_model, -0.5)
        self.current_lr = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        self.current_lr = lr
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr



class Trainner(object):
    def __init__(self, args, vocab):
        self.args = args
        self.vocab = vocab
        self.global_step = 0
        self.global_epoch = 0
        self.init_time = time.strftime('%Y%m%d_%H%M%S')
        self.device = args.device
        self.save_path = args.saved_model_root
        self.teacher_forcing_ratio = 1


    def train(self, model, loss_func, score_func, train_loader, dev_loader, teacher_forcing_ratio=-100, resume=False, exp_root=None):
        print(f'resume:{resume}')
        if exp_root is not None:
            self.exp_root = self.args.ckpt_root + exp_root
            self.tensorboard_root = self.exp_root + 'logs/'
            self.model_root = self.exp_root + 'saved_models/'
        else:
            self.exp_root = self.args.ckpt_root + self.init_time + '/'
            self.tensorboard_root = self.exp_root + 'logs/'
            self.model_root = self.exp_root + 'saved_models/'
            os.mkdir(self.exp_root)
            os.mkdir(self.tensorboard_root)
            os.mkdir(self.model_root)
        model.to(self.device)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        optim = t.optim.Adam([i for i in model.parameters() if i.requires_grad is True])
        optimizer = ScheduledOptim(optim, self.args.embedding_dim, 4000, n_current_steps=self.global_step)
        if resume:
            loaded = self._load(self.get_latest_cpath(), model)
            optimizer = loaded['optimizer']
            model = loaded['model']
            print(self.global_step, self.global_epoch)
        self.summary_writer = SummaryWriter(self.tensorboard_root)
        print(f'summary writer running in:')
        print(f'tensorboard --logdir {self.tensorboard_root}')
        self.summary_writer.add_embedding(model.embedding.weight.data, global_step=self.global_step)
        for epoch in range(self.args.epochs):
            self._train_epoch(model, optimizer, loss_func, score_func, train_loader, dev_loader)
            self.global_epoch += 1
            self.select_topk_model(self.args.num_model_tosave)
        self.summary_writer.close()
        print(f'DONE')

    def _train_epoch(self, model, optimizer, loss_func, score_func, train_loader, dev_loader):
        for data in tqdm(train_loader, desc='train step'):
            model.teacher_forcing = False
            self._train_step(model, optimizer, loss_func, data)
            # if self.global_step >= self.args.close_teacher_forcing_step:
            #     self.teacher_forcing_ratio = -100
            # else:
            #     self.teacher_forcing_ratio -= self.args.tf_ratio_decay_ratio
            self.teacher_forcing_ratio = -100
            if self.global_step % self.args.eval_every_step == 0:
                model.teacher_forcing_ratio = -100
                score = self._eval(model, loss_func, score_func, dev_loader)
                if self.global_step % self.args.save_every_step == 0:
                    self._save(model, self.global_epoch, self.global_step, optimizer, score)
            if self.global_step == 5000:
                self.summary_writer.add_embedding(model.embedding.weight.data, global_step=self.global_step)
    def _train_step(self, model, optimizer, loss_func, data):
        optimizer.zero_grad()
        train_loss = self._data2loss(model, loss_func, data)
        #train_loss.requires_grad = True
        train_loss.backward()
        model.embedding.weight.grad.data[0] = 0
        t.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5.0)
        optimizer.step_and_update_lr()

        self.summary_writer.add_scalar('loss/train_loss', train_loss.item(), self.global_step)
        #self.summary_writer.add_scalar('loss/train_report_loss', report_loss.item(), self.global_step)
        #self.summary_writer.add_scalar('teacher_forcing_ratio', model.teacher_forcing_ratio, self.global_step)
        #TODO add text writer for directly eval
        self.global_step += 1

    def _data2loss_rl(self, model, loss_func, data, score_func=None):
        context, title, context_lenths, title_lenths = [i.to(self.device) for i in data]
        token_id, prob_vector, sample_token_id, sample_prob_vector = model(context, context_lenths, title)
        #loss = loss_func(inputs = prob_vector, targets = title, target_lenth = title_lenths)
        loss, report_loss = loss_func(token_id, prob_vector, sample_token_id, sample_prob_vector, title, title_lenths)
        if score_func is None:
            return loss, report_loss
        else:
            score = score_func(token_id, title, self.args.eos_id)
            return loss, score, report_loss

    def _data2loss(self, model, loss_func, data, score_func=None):
        context, title, context_lenths, title_lenths = [i.to(self.device) for i in data]
        token_id, prob_vector, sample_token_id, sample_prob_vector = model(context, context_lenths, title)
        loss = loss_func(inputs = prob_vector, targets = title, target_lenth = title_lenths)
        #loss, report_loss = loss_func(token_id, prob_vector, sample_token_id, sample_prob_vector, title, title_lenths)
        if score_func is None:
            return loss
        else:
            score = score_func(token_id, title, self.args.eos_id)
            return loss, score


    def _eval(self, model, loss_func, score_func, dev_loader):
        eval_losses = []
        report_losses = []
        eval_scores = []
        model.eval()
        model.teacher_forcing = False
        with t.no_grad():
            for data in tqdm(dev_loader, desc='dev_step'):
                eval_loss, eval_score = self._data2loss(model, loss_func, data, score_func)
                eval_losses.append(eval_loss.item())
                #report_losses.append(report_loss.item())
                eval_scores.append(eval_score)
        eval_scores = np.mean(eval_scores)
        eval_losses = np.mean(eval_losses)
        eval_report_losses = np.mean(report_losses)
        self.summary_writer.add_scalar('loss/eval_loss', eval_losses, self.global_step)
        self.summary_writer.add_scalar('score/eval_score', eval_scores, self.global_step)
        #self.summary_writer.add_scalar('score/eval_report_loss', eval_report_losses, self.global_step)
        for i, v in model.named_parameters():
            self.summary_writer.add_histogram(i.replace('.', '/'), v.clone().cpu().data.numpy(), self.global_step)
        model.train()
        return eval_scores

    def _save(self, model, epoch, step, optimizer, score):
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        info = date_time + 'T' + str(score)
        path = os.path.join(self.model_root, info)
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
        file_name = os.listdir(self.model_root)
        latest = sorted(file_name, key=lambda x: x.split('T')[0], reverse=True)[0]
        return os.path.join(self.model_root, latest)

    def select_topk_model(self, k):
        file_name = os.listdir(self.model_root)
        remove_file = sorted(file_name, key=lambda x: x.split('T')[1], reverse=True)[k:]
        for i in remove_file:
            shutil.rmtree(os.path.join(self.model_root, i))
