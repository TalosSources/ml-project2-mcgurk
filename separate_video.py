person = "jad"
syllable = "va"

video_path = f"../dataset_to_process/{person}_{syllable}.mkv"
time_stamps_path = f"dataset/timestamps/{person}_{syllable}_timestamps.txt"

with open(time_stamps_path) as csvfile:
    splitted_lines = [l.split(sep=",") for l in csvfile.readlines()]
time_stamps = [(float(splits[0]) + float(splits[1]) / 30) for splits in splitted_lines]
print(time_stamps)

import os

for i, ts in enumerate(time_stamps):
    os.system(
        f"ffmpeg -ss {ts} -i {video_path} -frames:v 60 -vf scale=224:224 -c:v ffv1 dataset/syllables/{syllable}/{person}_{syllable}{i+1}.avi"
    )
