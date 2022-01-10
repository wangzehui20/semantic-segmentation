import torch
from model_zoo.segmodel import segmodel

if __name__ == '__main__':
    x = torch.zeros((1,3,224,224))
    encoder_name = 'mit'
    decoder_name = 'segformer_head'
    model = segmodel(encoder_name, decoder_name)
    pr = model(x)
    print(pr.shape)