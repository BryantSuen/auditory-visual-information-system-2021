from PIL import Image
from utils import read_video
import os
import imageio
from tqdm import tqdm

origin_dir = "./train/"
# origin_dir = "./test_offline/task1/"
target_dir = "./train_processed"

downsample_rate = 10

if __name__ == "__main__":
    for classes in tqdm(os.listdir(origin_dir)):
        class_dir = os.path.join(origin_dir, classes)
        class_dir_target = os.path.join(target_dir, classes)
        print(class_dir_target)
        os.makedirs(class_dir_target)
        cnt = 0
        for videos in os.listdir(class_dir):
            video, video_fps = read_video(os.path.join(class_dir, videos))
            for idx in range(len(video)):
                if idx % downsample_rate == 0:
                    file_name = videos[0:-4] + "_" + str(cnt) + ".png"
                    imageio.imwrite(os.path.join(class_dir_target, file_name), video[idx])
                    cnt += 1
            
