#from moviepy.editor import *
#
#video_path = '../dataset_to_process/ismail_ba.mkv'
#frame_stamps_path = 'time_stamps.txt'
#
#with open(frame_stamps_path) as csvfile:
#    splitted_lines = [l.split(sep=',') for l in csvfile.readlines()]
#frame_stamps = [splits[0] * 30 + splits[1] for splits in splitted_lines]
#
#video = VideoFileClip(video_path)
#
#for stamp in frame_stamps:
#    video.subclip()

video_path = '../dataset_to_process/ismail_ga.mkv'
frame_stamps_path = 'dataset/timestamps/ismail_ga_timestamps.txt'

with open(frame_stamps_path) as csvfile:
    splitted_lines = [l.split(sep=',') for l in csvfile.readlines()]
time_stamps = [(float(splits[0]) + float(splits[1]) / 30) for splits in splitted_lines]
print(time_stamps)

import os

for i, ts in enumerate(time_stamps):
    os.system(f"ffmpeg -ss {ts} -i {video_path} -frames:v 60 -vf scale=224:224 -c:v ffv1 dataset/ga/ismail_ga{i+1}.avi")