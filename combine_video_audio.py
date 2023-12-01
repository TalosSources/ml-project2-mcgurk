import os

for i in range(1, 16):
    os.system(f"ffmpeg -i dataset/ga/ismail_ga{i}.avi -i dataset/ba/ismail_ba{i}.avi -map 0:v -map 1:a -c copy dataset/ba_ga_da/ismail_ga_ba_da{i}.avi")