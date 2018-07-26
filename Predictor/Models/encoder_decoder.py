from Predictor.Blocks import RnnDecoder,RnnEncoder
import torch as t


class BaseModel(t.nn.Module):
    def __init__(self, matrix):
        super(BaseModel, self).__init__()
        self.embedding = matrix
        self.encoder = RnnEncoder()
        self.decoder = RnnDecoder()

    def forward(self, *input):
        for i in
