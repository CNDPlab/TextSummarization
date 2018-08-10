from configs import Config
from torch.utils.data import DataLoader
import torch as t
from Predictor.Utils import Vocab
import pickle as pk
from DataSets import DataSet, own_collate_fn
from Predictor.Models import EncoderDecoder
from Predictor.Utils.loss import masked_cross_entropy


args = Config()
train_set = DataSet('processed/train/')
dev_set = DataSet('processed/dev')
test_set = DataSet('processed/test/')
trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
devloader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)
testloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=own_collate_fn)

vocab = pk.load(open('Predictor/Utils/vocab.pkl', 'rb'))

model = EncoderDecoder(matrix=vocab.matrix, args=args)
optimizer = t.optim.Adam(model.parameters())
loss_func = masked_cross_entropy

global_step = 0
for i in range(args.epochs):
    for step, data in trainloader:





def prepare():
    pass



def train(**kwargs):
    args = Config()
    args.parse(kwargs)



def predict():
    pass
