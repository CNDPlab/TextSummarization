import torch as t
import time
import os
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
import shutil
import ipdb


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
        model.to(self.device)
        optimizer = t.optim.Adam(model.parameters())
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
            self._train_step(model, optimizer, loss_func, data)

            if self.global_step % self.args.eval_every_step == 0:
                score = self._eval(model, loss_func, score_func, dev_loader)
                if self.global_step % self.args.save_every_step == 0:
                    self._save(model, self.global_epoch, self.global_step, optimizer, score)
            if self.global_step == 500:
                self.summary_writer.add_embedding(model.embedding.weight.data, global_step=self.global_step)

    def _train_step(self, model, optimizer, loss_func, data):
        optimizer.zero_grad()
        train_loss = self._data2loss(model, loss_func, data)
        train_loss.backward()
        t.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5.0)
        optimizer.step()

        self.summary_writer.add_scalar('loss/train_loss', train_loss.item(), self.global_step)
        #TODO add text writer for directly eval
        self.global_step += 1

    def _data2loss(self, model, loss_func, data, score_func=None, ret_words=False):
        context, title, context_lenths, title_lenths = [i.to(self.device) for i in data]
        token_id, prob_vector = model(context)
        loss = loss_func(prob_vector, title, token_lenth, title_lenths)
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
            self.summary_writer.add_histogram(i.replace('.', '/') + '/grad', v.clone().cpu().data.numpy(), self.global_step)
        model.train()
        return eval_scores

    def write_sample_result_text(self, token_id, title):
        token_list = token_id.tolist()[0]
        title_list = title.tolist()[0]
        word_list = [self.vocab.from_id_token(word) for word in token_list]
        title_list = [self.vocab.from_id_token(word) for word in title_list]
        word_pre = ' '.join(word_list) + '\n' + ' '.joint(title_list)
        self.summary_writer.add_text('pre', word_pre, global_step=self.global_step)

    def _save(self, model, epoch, step, optimizer, score):
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        info = date_time + '_' + str(score)
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
        latest = sorted(file_name, key=lambda x: x.split('_')[0], reverse=True)[0]
        return os.path.join(self.model_root, latest)

    def select_topk_model(self, k):
        file_name = os.listdir(self.model_root)
        remove_file = sorted(file_name, key=lambda x: x.split('_')[1], reverse=True)[k:]
        for i in remove_file:
            shutil.rmtree(os.path.join(self.model_root, i))
