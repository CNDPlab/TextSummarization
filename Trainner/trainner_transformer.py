import torch as t
import time
import os
from tensorboardX import SummaryWriter
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


class Trainner_transformer(object):
    def __init__(self, args, vocab):
        self.args = args
        self.vocab = vocab
        self.global_step = 0
        self.global_epoch = 0
        self.init_time = time.strftime('%Y%m%d_%H%M%S')
        self.device = args.device
        self.save_path = args.saved_model_root

    def train(self, model, loss_func, score_func, train_loader, dev_loader, resume, exp_root=None):
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
        print(f'exp_root{self.exp_root}')
        model = t.nn.DataParallel(model).cuda()
        optimizer = t.optim.Adam([i for i in model.parameters() if i.requires_grad is True])
        optim = ScheduledOptim(optimizer, self.args.embedding_dim, 4000, n_current_steps=self.global_step)
        if resume:
            loaded = self._load(self.get_best_cpath(), model)
            self.global_step = loaded['step']
            self.global_epoch = loaded['epoch']
            optim = loaded['optim']
            model = loaded['model']
            print(self.global_step, self.global_epoch)
        self.summary_writer = SummaryWriter(self.tensorboard_root)
        print(f'summary writer running in:')
        print(f'tensorboard --logdir {self.tensorboard_root}')

        for epoch in range(self.args.epochs):
            self._train_epoch(model, optim, loss_func, score_func, train_loader, dev_loader)
            self.global_epoch += 1
            self.select_topk_model(self.args.num_model_tosave)
        self.summary_writer.close()
        print(f'DONE')

    def _train_epoch(self, model, optim, loss_func, score_func, train_loader, dev_loader):
        for data in tqdm(train_loader, desc='train step'):
            self._train_step(model, optim, loss_func, data)

            if self.global_step % self.args.eval_every_step == 0:
                score = self._eval(model, loss_func, score_func, dev_loader)
                if self.global_step % self.args.save_every_step == 0:
                    self._save(model, self.global_epoch, self.global_step, optim, score)

    def _train_step(self, model, optim, loss_func, data):
        optim.zero_grad()
        train_loss = self._data2loss(model, loss_func, data)
        train_loss.backward()
        t.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5.0)
        optim.step_and_update_lr()
        self.summary_writer.add_scalar('loss/train_loss', train_loss.item(), self.global_step)
        self.summary_writer.add_scalar('lr', optim.current_lr, self.global_step)
        self.global_step += 1

    def _data2loss(self, model, loss_func, data, score_func=None, ret_words=False):
        context, title, context_lenths, title_lenths = [i.cuda() for i in data]
        token_id, prob_vector = model(context, title)
        loss = loss_func(prob_vector, title)
        if score_func is None:
            if not ret_words:
                return loss
            else:
                return loss, token_id, title
        else:
            if not ret_words:
                score = score_func(token_id, title, self.args.eos_id)
                return loss, score
            else:
                score = score_func(token_id, title, self.args.eos_id)
                return loss, score, token_id, title

    def _eval(self, model, loss_func, score_func, dev_loader):
        eval_losses = []
        eval_scores = []
        model.eval()
        with t.no_grad():
            for data in tqdm(dev_loader, desc='dev_step'):
                eval_loss, eval_score, token_id, title = self._data2loss(model, loss_func, data, score_func, ret_words=True)
                eval_losses.append(eval_loss.item())
                eval_scores.append(eval_score)
        self.write_sample_result_text(token_id, title)
        eval_scores = np.mean(eval_scores)
        eval_losses = np.mean(eval_losses)
        self.summary_writer.add_scalar('loss/eval_loss', eval_losses, self.global_step)
        self.summary_writer.add_scalar('score/eval_score', eval_scores, self.global_step)
        for i, v in model.named_parameters():
            self.summary_writer.add_histogram(i.replace('.', '/'), v.clone().cpu().data.numpy(), self.global_step)
        model.train()
        return eval_scores

    def write_sample_result_text(self, token_id, title):
        token_list = token_id.data.tolist()[0]
        title_list = title.data.tolist()[0]
        word_list = [self.vocab.from_id_token(word) for word in token_list]
        title_list = [self.vocab.from_id_token(word) for word in title_list]
        word_pre = ' '.join(word_list) + '---' + ' '.join(title_list)
        self.summary_writer.add_text('pre', word_pre, global_step=self.global_step)

    def _save(self, model, epoch, step, optim, score):
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        info = date_time + 'T' + str(score)
        path = os.path.join(self.model_root, info)
        if not os.path.exists(path):
            os.mkdir(path)
        t.save({'epoch': epoch,
                'step': step,
                'optim': optim
                }, os.path.join(path, 'trainer_state'))
        t.save(model.state_dict(), os.path.join(path, 'model'))

    def _load(self, path, model):
        resume_checkpoint = t.load(os.path.join(path, 'trainer_state'))
        model.load_state_dict(t.load(os.path.join(path, 'model')))
        return {'epoch': resume_checkpoint['epoch'],
                'step': resume_checkpoint['step'],
                'optim': resume_checkpoint['optim'],
                'model': model}

    def get_best_cpath(self):
        file_name = os.listdir(self.model_root)
        latest = sorted(file_name, key=lambda x: x.split('T')[1], reverse=True)[0]
        print(f'best model loaded is {latest}')
        return os.path.join(self.model_root, latest)

    def select_topk_model(self, k):
        file_name = os.listdir(self.model_root)
        remove_file = sorted(file_name, key=lambda x: x.split('T')[1], reverse=True)[k:]
        for i in remove_file:
            shutil.rmtree(os.path.join(self.model_root, i))
