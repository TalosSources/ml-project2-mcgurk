import os

auditory = "ba"
visual = "fa"
mg_expected = "va"
person = "ismail"

for i in range(21, 22):
    os.system(
        f"ffmpeg -i dataset/syllables/{visual}/{person}_{visual}{i}.avi -i dataset/syllables/{auditory}/{person}_{auditory}{i}.avi -map 0:v -map 1:a -c copy dataset/mcgurk/{auditory}_{visual}_{mg_expected}/{person}_{auditory}_{visual}_{mg_expected}{i}.avi"
    )
    # os.system(f"""ffmpeg -i dataset/ba/jad_ba{i}.avi -i dataset/ga/jad_ga{i}.avi -filter_complex "[1:a]adelay=200[a1];[0:a][a1]amix[a]" -map "[a]" -map 0:v -c:v libx264 -c:a aac -strict experimental -shortest dataset/ga_ba_da/jad_ga_ba_da{i}.avi""")
