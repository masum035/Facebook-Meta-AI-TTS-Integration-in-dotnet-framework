import os

import torch
import torch.nn as nn

from Loading_Chekpoints import TextMapper
from vits import utils
from vits.models import SynthesizerTrn

# model = torch.load("ben//G_100000.pth", map_location=torch.device('cpu'))

ckpt_dir = os.path.join("ben")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Run inference with {device}")

vocab_file = f"{ckpt_dir}/vocab.txt"
config_file = f"{ckpt_dir}/config.json"
assert os.path.isfile(config_file), f"{config_file} doesn't exist"

hps = utils.get_hparams_from_file(config_file)
text_mapper = TextMapper(vocab_file)
net_g = SynthesizerTrn(
    len(text_mapper.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
net_g.to(device)
_ = net_g.eval()

g_pth = f"{ckpt_dir}/G_100000.pth"
print(f"load {g_pth}")

model, optimizer, learning_rate, iteration = utils.load_checkpoint(g_pth, net_g, None)

# model.load_state_dict(torch.load('path_to_your_model.pth'))
# model.eval()

if isinstance(model.features[0], nn.Conv2d):
    first_conv_layer = model.features[0]
    input_channels = first_conv_layer.in_channels
    print("Input channels: ", input_channels)
