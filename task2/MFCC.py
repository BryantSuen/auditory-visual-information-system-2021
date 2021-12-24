# -*- coding: utf-8 -*-
'''
功能：可直接调用本文件中的 MFCC(wav_file) 函数来将一个.wav文件提取出其特征MFCC
'''
import numpy as np
import scipy.io.wavfile
from matplotlib import pyplot as plt
from scipy.fftpack import dct


N_FFT = 512
TIME_LEN = 3.5 # 每一个wav文件的采样时间
FRAME_SIZE = 0.025
FRAME_STRIDE = 0.01
# 1.采样：
def sampling(wav_file):
    #采样：
    sample_rate, signal = scipy.io.wavfile.read(wav_file)
    #读取前3.5s
    original_signal = signal[0:int(TIME_LEN*sample_rate)]
    return sample_rate, original_signal
# 2.预加重
def pre_emphasis(signal):
    miu = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - miu * signal[:-1])
    return emphasized_signal
# 3. 分帧
def framing(signal,sample_rate):
    FRAME_SIZE = 0.025  # 帧长25ms
    FRAME_STRIDE = 0.01 # 上一帧的开始和下一帧的开始间隔10ms，即拥有15ms的重叠
    frame_length = int(round(FRAME_SIZE*sample_rate)) # 单位转化为采样点数
    frame_step = int(round(FRAME_STRIDE*sample_rate)) 
    signal_length = len(signal)
    num_frames = int(np.ceil( (float((signal_length - frame_length)) / frame_step) ))# 帧数量
    frames = np.zeros([num_frames,frame_length])

    new_signal = np.append(signal, np.zeros((num_frames * frame_step + frame_length - signal_length))) #不够的部分补0
    for id in range(num_frames):
        frames[id] = new_signal[id * frame_step : id * frame_step + frame_length]
    return frames,new_signal
# 4.加窗，fft
def hamming_and_fft(frames,N_fft):
    [frame_number,frame_length] = frames.shape
    frames_new = frames * np.hamming(frame_length)
    mag_frames = np.abs(np.fft.rfft(frames_new, N_fft))
    pow_frames = (1.0 / N_fft) * (mag_frames ** 2)
    return pow_frames
# 5.mel滤波
def mel_filter(frames,N_fft,sample_rate):
    low_freq_mel = 0
    filter_num = 40
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, filter_num + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((N_fft + 1) * hz_points / sample_rate)
    fbank = np.zeros((filter_num, int(np.floor(N_fft / 2 + 1))))
    for m in range(1, filter_num + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    return filter_banks
# 6. 得到mfcc
def getmfcc(filter_banks):
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    cep_lifter =22
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    return mfcc

# 整体函数
def MFCC(wav_file):
    # 输入
    # wav_file : 需要读取的wav文件路径
    # 输出
    # mfcc : 提取出的mfcc矩阵，其中不同行 为不同帧的mfcc系数
    sample_rate, original_signal = sampling(wav_file)
    signal_emphasized = pre_emphasis(original_signal)
    frames, signal_frame = framing(signal_emphasized,sample_rate) 
    
    pow_frames = hamming_and_fft(frames,N_FFT)
    filter_banks = mel_filter(pow_frames,N_FFT,sample_rate)
    mfcc = getmfcc(filter_banks)
    return mfcc
