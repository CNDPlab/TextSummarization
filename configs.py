import warnings


class Config(object):
    datas_root = 'Datas/'
    raw_folder = 'Datas/raw/'
    middle_folder = 'Datas/middle/'
    processed_folder = 'Datas/processed/'
    saved_vocab = 'Predictor/Utils/vocab.pkl'
    tensorboard_root = 'ckpt/logs/'
    saved_model_root = 'ckpt/saved_models/'
    model_name = 'EncoderDecoder'
    resume = False
    device = 'cuda:0'
    embedding_dim = 128
    epochs = 20
    beam_size = 3
    batch_size = 32
    padding_idx = 0
    hidden_size = 128
    dropout = 0.5
    init_tf_ratio = 1
    tf_ratio_decay_ratio = 0.9999
    close_teacher_forcing_step = 50000
    num_layers = 2
    num_model_tosave = 3
    sos_id = None
    eos_id = None
    decoding_max_lenth = 50
    eval_every_step = 500
    save_every_step = 4000
    unk_ratio = 0.3

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print('__', k, getattr(self, k))