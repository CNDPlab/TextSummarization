import warnings


class Config(object):
    datas_root = 'Datas/'
    raw_folder = 'Datas/LCSTS2.0/'
    middle_folder = 'Datas/middle/'
    processed_folder = 'Datas/processed/'
    nlpcc_middle = 'Datas/nlpcc_middle/'
    nlpcc_processed = 'Datas/nlpcc_processed/'
    saved_vocab = 'Predictor/Utils/vocab.pkl'
    ckpt_root = 'ckpt/'
    saved_model_root = 'ckpt/saved_models/'
    model_name = 'UniversalTransformer'
    resume = False
    device = 'cuda:1'
    exp_root = None
    embedding_dim = 256
    encoder_max_lenth = 500
    max_step_lenth = 7
    epochs = 50
    beam_size = 10
    num_head = 8
    batch_size =256
    padding_idx = 0
    hidden_size = 64
    dropout = 0.1
    num_layers = 1
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