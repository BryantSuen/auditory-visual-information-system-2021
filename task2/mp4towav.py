from moviepy.editor import *
import os

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
	else:
		print("---  There is this folder!  ---")

def mp4towav(src,dst):
    video = VideoFileClip(src)
    audio = video.audio
    audio.write_audiofile(dst)

root = "./task2/train_audio"
mp4_root = "./train"

if __name__ == "__main__":
    for id in range(1,21):
        path = root + "/ID" + str(id)
        mkdir(path)
        mp4_folder = mp4_root + "/ID" + str(id)
        for file in os.listdir(mp4_folder):
            mp4_path = mp4_folder + "/" + file
            wav_path = path + "/" + file[0:-4] + ".wav"
            mp4towav(mp4_path,wav_path) 
