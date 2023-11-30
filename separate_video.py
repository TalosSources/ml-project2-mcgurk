from moviepy.editor import *

video_path = 'dataset/ismail_ba.mkv'
frame_stamps_path = 'time_stamps.txt'

with open(frame_stamps_path) as csvfile:
    splitted_lines = [l.split(sep=',') for l in csvfile.readlines()]
frame_stamps = [splits[0] * 30 + splits[1] for splits in splitted_lines]

video = VideoFileClip(video_path)

for stamp in frame_stamps:
    video.subclip
