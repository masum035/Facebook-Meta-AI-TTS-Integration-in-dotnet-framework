import os
import string
import time

import numpy
import torch

from Loading_Chekpoints import TextMapper
from vits import utils
from vits.models import SynthesizerTrn
from scipy.io.wavfile import write


def Text_To_ByteArray_Conversion(txt: string):
    start = time.time()
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

    _ = utils.load_checkpoint(g_pth, net_g, None)

    txt = txt.lower()
    txt = text_mapper.filter_oov(txt)
    # txt = preprocess_text(txt, text_mapper, hps, lang=LANGUAGE)
    stn_tst = text_mapper.get_text(txt, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        hyp = net_g.infer(
            x_tst, x_tst_lengths, noise_scale=.667,
            noise_scale_w=0.8, length_scale=1.0
        )[0][0, 0].cpu().float().numpy()

    end = time.time()

    print('Just to generate byte array, took Time (in seconds): ', end - start)
    print(hyp.ndim, " dimensional array is audio output")
    print("------------------------------------------------------------")
    # print(hyp)
    return hyp, hps.data.sampling_rate


audio_array, sampling_rate = Text_To_ByteArray_Conversion(
    txt=" প্রাকৃতিক রূপবৈচিত্র্যে ভরা আমাদের এই Bangladesh। এই দেশে পরিচিত অপরিচিত অনেক পর্যটক-আকর্ষক Place আছে। ")


def ByteArray_To_Audio_Conversion(array: numpy.ndarray, sample_rate: int):
    start = time.time()
    wav_file_path = os.path.join("audio_folder//meta_ai_TTS_example_02.wav")
    write(filename=wav_file_path, rate=sample_rate, data=array)
    end = time.time()
    print('Just to generate audio file, took Time (in seconds): ', end - start)


ByteArray_To_Audio_Conversion(audio_array, sampling_rate)
