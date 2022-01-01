import os
import numpy as np

from utils import read_audio
from tqdm import tqdm

origin_dir = "./train/"
# origin_dir = "./test_offline/task1/"
target_dir = "./train_audio"

downsample_interval = 40000
stride = 20000

if __name__ == "__main__":
    for classes in tqdm(os.listdir(origin_dir)):
        class_dir = os.path.join(origin_dir, classes)
        class_dir_target = os.path.join(target_dir, classes)
        print(class_dir_target)
        os.makedirs(class_dir_target)

        for audios in os.listdir(class_dir):
            audio = read_audio(os.path.join(class_dir, audios))
            audio_len = audio.shape[0]
            cnt = 0
            while True:
                if cnt * stride + downsample_interval <= audio_len:
                    file_name = audios[0:-4] + "_" + str(cnt) + ".npy"
                    np.save(os.path.join(class_dir_target, file_name), audio[cnt * stride: cnt * stride + downsample_interval])
                    cnt += 1
                else: 
                    break
            # for idx in range(len(video)):
            #     if idx % downsample_rate == 0:
            #         file_name = videos[0:-4] + "_" + str(cnt) + ".png"
            #         imageio.imwrite(os.path.join(class_dir_target, file_name), video[idx])
            #         cnt += 1