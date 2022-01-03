from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import torch
from utils import read_audio

def separator_3mix(speech_path):
    # for 3 persons
    model = separator.from_hparams(source="speechbrain/sepformer-wsj03mix", savedir='pretrained_models/sepformer-wsj03mix')
    # for custom file, change path
    est_sources = model.separate_file(path=speech_path)
    return est_sources
if __name__ == '__main__':
    speech = read_audio("./test_offline/task3/combine001.mp4")
    print(speech.shape)
    srcs = separator_3mix(speech[:, 0])
    print(srcs.shape)