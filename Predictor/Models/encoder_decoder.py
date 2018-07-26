from Predictor.Blocks import RnnDecoder,RnnEncoder
import torch as t


class BaseModel(t.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.encoder = RnnEncoder()
        self.decoder = RnnDecoder()

    def forward(self, *input):
        pass

