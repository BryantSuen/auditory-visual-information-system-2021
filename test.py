from ctypes import util
import numpy as np
import soundfile as sf
import os,json
import utils
import nussl
import torch
from tqdm import tqdm

from task1 import classify_video
from task2 import classify_audio
from task3 import separator_3mix
import models

def test_task1(video_path):
    # 测试1
    result_dict = {}
    model = models.model_task1
    state_dict = torch.load("./models/task1_resnet34.pkl")
    model.load_state_dict(state_dict)
    for file_name in tqdm(os.listdir(video_path)):
        ## 读取MP4文件中的视频,可以用任意其他的读写库
        # video_frames,video_fps = utils.read_video(os.path.join(video_path,file_name))
        result = classify_video(os.path.join(video_path,file_name), model)

        ## 做一些处理
        # print('video_frames have shape of:',video_frames.shape, 'and fps of:',video_fps)

        ## 返回一个ID 
        result_dict[file_name]=utils.ID_dict[result + 1]

    return result_dict

def test_task2(wav_path):
    # 测试2
    result_dict = {}
    model = models.model_task2
    state_dict = torch.load("./models/task2_resnet34.pkl")
    model.load_state_dict(state_dict)
    for file_name in tqdm(os.listdir(wav_path)):
        ## 读取WAV文件中的音频,可以用任意其他的读写库
        # audio_trace = utils.read_audio(os.path.join(wav_path,file_name),sr=44100)
        result = classify_audio(os.path.join(wav_path, file_name), model)

        ## 做一些处理
        # print('audio_trace have shape of:',audio_trace.shape,'and sampling rate of: 44100')

        ## 返回一个ID
        result_dict[file_name]=utils.ID_dict[result + 1]

    return result_dict

def test_task3(video_path,result_path):
    # 测试2
    if os.path.isdir(result_path):
        print('warning: using existed path as result_path')
    else:
        os.mkdir(result_path)
    for file_name in os.listdir(video_path):
        ## 读MP4中的图像和音频数据，例如：
        idx = file_name[-7:-4]  # 提取出序号：001, 002, 003.....

        video_frames,video_fps= utils.read_video(os.path.join(video_path,file_name))
        audio_trace = utils.read_audio(os.path.join(video_path,file_name),sr=44100)
        ## save tmp file
        if not os.path.exists("./tmp"):
            os.mkdir("./tmp")
        sf.write("./tmp/tmp.wav", audio_trace, 44100)

        result = separator_3mix("./tmp/tmp.wav")
        ## 做一些处理
        print('video_frames have shape of:',video_frames.shape, 'and fps of:',video_fps)
        print('audio_trace have shape of:',audio_trace.shape,'and sampling rate of: 44100')

        audio_left   = result[:, :, 0].T.numpy()
        audio_middle   = result[:, :, 1].T.numpy()
        audio_right   = result[:, :, 2].T.numpy()

        ## 输出结果到result_path
        sf.write(os.path.join(result_path,idx+'_left.wav'),   audio_left, 8000)
        sf.write(os.path.join(result_path,idx+'_middle.wav'), audio_middle, 8000)
        sf.write(os.path.join(result_path,idx+'_right.wav'),  audio_right, 8000)


if __name__=='__main__':

    # ## testing task1
    # with open('./test_offline/task1_gt.json','r') as f:
    #     task1_gt = json.load(f)
    # task1_pred = test_task1('./test_offline/task1')
    # # print("*********************")
    # # print(task1_pred)
    # task1_acc = utils.calc_accuracy(task1_gt,task1_pred)
    # print('accuracy for task1 is:',task1_acc)   

    # # ## testing task2
    # with open('./test_offline/task2_gt.json','r') as f:
    #     task2_gt = json.load(f)
    # task2_pred = test_task2('./test_offline/task2')
    # task2_acc = utils.calc_accuracy(task2_gt,task2_pred)
    # print('accuracy for task2 is:',task2_acc)   

    # # testing task3
    test_task3('./test_offline/task3','./test_offline/task3_estimate')
    task3_SISDR_blind = utils.calc_SISDR('./test_offline/task3_gt','./test_offline/task3_estimate',permutaion=True)  # 盲分离
    print('strength-averaged SISDR_blind for task3 is:',task3_SISDR_blind)
    task3_SISDR_match = utils.calc_SISDR('./test_offline/task3_gt','./test_offline/task3_estimate',permutaion=False) # 定位分离
    print('strength-averaged SISDR_match for task3 is: ',task3_SISDR_match)

