import torch as t


encoder_output = t.randn((32, 20, 12))
decoder_step_output = t.randn((32, 12))



corelation_score = t.bmm(encoder_output,decoder_step_output.unsqueeze(-1))
# shape : [32, 20, 1]
attention_score = t.nn.functional.softmax(corelation_score, dim=-2)
# shape : [32, 20, 1]
attention_vector = t.bmm(encoder_output.transpose(-1,-2), attention_score)