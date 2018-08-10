class Config(object):
    raw_folder = 'raw/'
    raw_file = 'raw/df.json'
    middle_folder = 'middle/'
    processed_folder = 'processed/'
    saved_vocab = 'Predictor/Utils/vocab.pkl'
    tensorboard_root = 'ckpt/logs/'
    saved_model_root = 'ckpt/saved_models/'
    model_name = 'encoder_decoder'
    #TODO check device using
    device = 'gpu:0'
    embedding_dim = 128
    epochs = 20

    batch_size = 32
    padding_idx = 0
    hidden_size = 128
    dropout = 0.5
    num_layers = 2
    sos_id = 2
    eos_id = 3
    decoding_max_lenth = 8

    eval_every_step = 100


    def parse(self):
        #TODO add parse func
        pass