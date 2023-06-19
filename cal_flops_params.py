
import torch
import getters
from thop import profile, clever_format

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

architecture = 'UnetEdge'

init_params = dict(
    encoder_name='efficientnet-b3',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
    decoder_attention_type='scse'
)
inp = torch.randn((1, 3, 128, 128)).to(device)
model = getters.get_model(architecture=architecture, init_params=init_params).to(device)
flops, params = profile(model, inputs=(inp, ))
flops, params = clever_format([flops, params], "%.3f")

print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
print("---|---|---")
print(
    "%s | %s | %s" % (architecture, params, flops)
)