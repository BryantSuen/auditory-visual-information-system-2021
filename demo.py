# import librosa
# import numpy as np
# from python_speech_features import fbank

# def normalize_frames(m,Scale=True):
#     if Scale:
#         return (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 2e-12)
#     else:
#         return (m - np.mean(m, axis=0))

# audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
# filter_banks, energies = fbank(audio, samplerate=sample_rate, nfilt=40, winlen=0.025)
# filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))
# feature = normalize_frames(filter_banks, Scale=False)

import numpy as np
b = np.load("./train_audio/ID1/014_0.npy")
print(b.shape)

# # from torchvision import models, transforms
# from torchvision.models import resnet34
# from torchaudio.models  import DeepSpeech
# import torch.nn as nn
# import torch
# import torchaudio
# import matplotlib.pyplot as plt
# import python_speech_features
# import librosa

# from models import model_task1
# # from dataset import video_dataset
# from utils import read_video, read_audio

# # model = DeepSpeech(20)
# # print(model)
# # a = torch.tensor([[1,2,3],[4,5,0]])
# # b = a.argmax(dim = 1)
# # print(b)

# a = resnet34()

# print(a)
## demo for python_speech_features
# a = read_audio("./train/ID1/014.mp4")
# mfcc = librosa.feature.mfcc(a, 44100)
# print(mfcc.shape)


# result_dict = {}
# model = model_task1
# state_dict = torch.load("./models/task1resnet18.pkl")
# model.load_state_dict(state_dict)
# a = torch.ones([3,4])
# output = model(a)
# aaa = read_audio("./train/ID1/014.mp4")
# print(aaa.shape) # 443117, 1
# waveform, sample_rate = torchaudio.load("./task2/train_audio/ID1/017.wav")
# print(waveform[0])
# print("Shape of waveform: {}".format(waveform.size()))
# print("Sample rate of waveform: {}".format(sample_rate))

# specgram = torchaudio.transforms.MelSpectrogram()(waveform)

# print("Shape of spectrogram: {}".format(specgram.size()))

# plt.figure()
# p = plt.imshow(specgram.log2()[0,:,:].detach().numpy(), cmap='gray')
# plt.pause(5)

# mobilenet = models.mobilenet

# a = torch.tensor([1,2,3,4,5,4])
# a[1] = 99
# print(a)
# resnet = models.resnet101(pretrained=True)

# resnet.fc = nn.Linear(in_features=2048, out_features= 10, bias = True)
# print(resnet.fc)
# video, video_fps = read_video("./train/ID1/014.mp4")

# print(video.shape)
# print(len(video))

# preprocess = transforms.Compose([
#     transforms.ToTensor()
#     ]
# )


# data = torch.stack([preprocess(video[0]), preprocess(video[1])], dim=0)
# print("data size: ")
# resnet.eval()
# out = resnet(data)

# print(out.size())
# print(out)

# criterion = nn.CrossEntropyLoss()

# loss = criterion(out, out)
# print("okkkkk")
# print(loss)
# a = "asasa.mp4"
# b = a + str(10)