import torch as t
from tensorboardX import SummaryWriter



a = t.Tensor([[0.1,0.6,0.3],[0.2,0.4,0.4],[0.7,0.1,0.2]])




writer = SummaryWriter('ckpt/')

for i in range(1000):
    writer.add_image('img', a, i)
    writer.add_scalar('test', t.Tensor([1]), i)

writer.close()

