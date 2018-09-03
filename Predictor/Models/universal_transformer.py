import torch as t
import numpy as np


class UniversalTransformer(t.nn.Module):
    def __init__(self, args, matrix):
        super(UniversalTransformer, self).__init__()
        self.args = args
        self.matrix = matrix
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, ):
        pass


class Encoder(t.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.word_embedding = t.nn.Embedding()
        self.position_embedding = t.nn.Embedding()
        self.step_embedding = t.nn.Embedding()
        self.encoder_block = EncoderBlock()

    def add_position_feature(self, inputs):
        pass

    def add_step_feature(self, inputs, step):
        pass

    def forward(self, word_feature, step):
        net = self.word_embedding(word_feature)


        for step in range(1, self.num_step + 1):
            net = self.add_position_feature(net)
            net = self.add_step_feature(net, step)
            net = self.encoder_block(net)
        return net


class EncoderBlock(t.nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()
        self.multi_head_self_attetnion = MultiHeadSelfAttention()
        self.drop_out = t.nn.Dropout()
        self.layer_normalization1 = LayerNormalization()
        self.transition_function = TransitionFunction()
        self.layer_normalization2 = LayerNormalization()

    def forward(self, inputs, inputs_mask):
        net = self.multi_head_self_attetnion(inputs, inputs, inputs)
        net = net + inputs
        net = self.drop_out(net)
        normallized = self.layer_normalization1(net)
        normallized = self.transition_function(normallized)
        net = net + normallized
        net = self.drop_out(net)
        #TODO share layer_normalization?
        net = self.layer_normalization2(net)
        return net


class Decoder(t.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.word_embedding = t.nn.Embedding()
        self.position_embedding = t.nn.Embedding()
        self.step_embedding = t.nn.Embedding()
        self.decoder_block = ()

    def forward(self,):
        pass


class DecoderBlock(t.nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()


class StepEmbedding(t.nn.Module):
    def __init__(self):
        super(StepEmbedding, self).__init__()

    def forward(self):
        pass


class PositionEmbedding(t.nn.Module):
    def __init__(self):
        super(PositionEmbedding, self).__init__()

    def forward(self):
        pass


class MultiHeadSelfAttention(t.nn.Module):
    def __init__(self):
        super(MultiHeadSelfAttention, self).__init__()

    def forward(self):
        pass

class SelfAttention(t.nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self):
        pass


class LayerNormalization(t.nn.Module):
    def __init__(self):
        super(LayerNormalization, self).__init__()

    def forward(self):
        pass


class TransitionFunction(t.nn.Module):
    def __init__(self):
        super(TransitionFunction, self).__init__()

    def forward(self):
        pass

