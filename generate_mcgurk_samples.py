import os

dataset = 'dataset/train'

experiments = [
    (('ba', 'ga', 'da'), 0),
    (('ba', 'fa', 'va'), 0),
    (('ga', 'ba', 'bga'), 300),
]



for ((a, v, mg), dt) in experiments:
    exp = f"{a}_{v}_{mg}"
    for person in ('ismail', 'jad', 'olena'):
        for i in range(1, 20):
            if mg != 'bga':
                os.system(f"""ffmpeg -hide_banner -loglevel error -i dataset/train/{v}/{v}_{person}_{i}.avi -i dataset/train/{a}/{a}_{person}_{i}.avi -map 0:v -map 1:a -frames:v 40 -c copy dataset/mcgurk/{exp}/{exp}_{person}_{i}.avi""")
            else:
                #os.system(f"""ffmpeg -hide_banner -loglevel error -i dataset/train/{v}/{v}_{person}_{i}.avi -i dataset/train/{a}/{a}_{person}_{i}.avi -filter_complex "[1:a]adelay={dt}[a1];[a1][0:v]concat=n=2:v=1:a=0[v]" -map "[v]" -frames:v 40 -c:v libx264 -c:a aac -strict experimental -shortest dataset/mcgurk/{exp}/{exp}_{person}_{i}.avi""")
                os.system(f"""ffmpeg -hide_banner -loglevel error -i dataset/train/{v}/{v}_{person}_{i}.avi -i dataset/train/{a}/{a}_{person}_{i}.avi -filter_complex "[1:a]adelay={dt}[a]" -map "[a]" -map 0:v -frames:v 40 -c:v libx264 -c:a aac -strict experimental -shortest dataset/mcgurk/{exp}/{exp}_{person}_{i}.avi""")
                #os.system(f"""ffmpeg -hide_banner -loglevel error -i dataset/train/{v}/{v}_{person}_{i}.avi -i dataset/train/{a}/{a}_{person}_{i}.avi -filter_complex "[1:a]adelay={dt}[a1];[0:a][a1]amix[a]" -map "[a]" -map 0:v -frames:v 40 -c:v libx264 -c:a aac -strict experimental -shortest dataset/mcgurk/{exp}/{exp}_{person}_{i}.avi""")
