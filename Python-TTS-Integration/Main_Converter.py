import os
import string
import time

import torch

from Loading_Chekpoints import TextMapper
from vits import utils
from vits.models import SynthesizerTrn


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
    print(hyp)


Text_To_ByteArray_Conversion(
    txt=" প্রাকৃতিক রূপবৈচিত্র্যে ভরা আমাদের এই বাংলাদেশ। এই দেশে পরিচিত অপরিচিত অনেক পর্যটক-আকর্ষক স্থান আছে। এর মধ্যে প্রত্নতাত্ত্বিক নিদর্শন, ঐতিহাসিক মসজিদ এবং মিনার, পৃথিবীর দীর্ঘতম প্রাকৃতিক সমুদ্র সৈকত, পাহাড়, অরণ্য ইত্যাদি অন্যতম। এদেশের প্রাকৃতিক সৌন্দর্য পর্যটকদের মুগ্ধ করে। বাংলাদেশের প্রত্যেকটি এলাকা বিভিন্ন স্বতন্ত্র্র বৈশিষ্ট্যে বিশেষায়িত । বাংলাদেশ দক্ষিণ এশিয়ার উত্তর পূর্ব অংশে অবস্থিত। বাংলাদেশের উত্তর সীমানা থেকে কিছু দূরে হিমালয় পর্বতমালা এবং দক্ষিণে বঙ্গোপসাগর। পশ্চিমে ভারতের পশ্চিমবঙ্গ, পূর্বে ভারতের ত্রিপুরা, মিজোরাম রাজ্য এবং মায়ানমারের পাহাড়ী এলাকা। অসংখ্য নদ-নদী পরিবেষ্টিত বাংলাদেশ প্রধানত সমতল ভূমি। দেশের উল্লেখযোগ্য নদ-নদী হলো- পদ্মা, ব্রহ্মপুত্র, সুরমা, কুশিয়ারা, মেঘনা ও কর্ণফুলী। একেকটি অঞ্চলের প্রাকৃতিক সৌন্দর্য ও খাদ্যাভ্যাস বিভিন্ন ধরনের। বাংলাদেশ রয়েল বেঙ্গল টাইগারের দেশ যার বাস সুন্দরবনে। এছাড়াও এখানে রয়েছে লাল মাটি দিয়ে নির্মিত মন্দির। এদেশে উল্লেখযোগ্য পর্যটন এলাকার মধ্যে রয়েছে: শ্র্রীমঙ্গল, যেখানে মাইলের পর মাইল জুড়ে রয়েছে চা বাগান। প্রত্নতাত্ত্বিক নিদর্শনের স্থানগুলোর মধ্যে রয়েছে–ময়নামতি, মহাস্থানগড় এবং পাহাড়পুর। রাঙ্গামাট, কাপ্তাই এবং কক্সবাজার প্রাকৃতিক দৃশ্যের জন্য খ্যাত। সুন্দরবনে আছে বন্য প্রাণী এবং পৃথিবীখ্যাত ম্যানগ্রোভ ফরেস্ট এ বনাঞ্চলে অবস্থিত । ")
