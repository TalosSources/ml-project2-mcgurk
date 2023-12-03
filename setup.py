dataset_url = (
    "https://drive.google.com/file/d/1fkgTer7KxEkoFu0MmssUn7iYa1SSdAHN/view?usp=sharing"
)
output = "mcgurk_uncompressed.zip"

# Download the dataset
import gdown
import os.path

if not os.path.isfile(output):
    gdown.download(dataset_url, output, quiet=False, fuzzy=True)

# Unzip the dataset
import zipfile

with zipfile.ZipFile(output, "r") as zip_ref:
    zip_ref.extractall("./dataset/raw/videos/")

# Remove the zip file
# import os
# os.remove(output)

# Generate samples from the raw dataset
# Pair csv files with mkv videos of the same name
import os
import glob

# Get all csv files
csv_files = glob.glob("./dataset/raw/timestamps/*.csv")
# Get all mkv files
mkv_files = glob.glob("./dataset/raw/videos/*.mkv")

# Pair csv files with mkv videos of the same name
csv_mkv_pairs = []
for csv_file in csv_files:
    csv_file_name = csv_file.split("/")[-1].split(".")[0]
    for mkv_file in mkv_files:
        mkv_file_name = mkv_file.split("/")[-1].split(".")[0]
        if csv_file_name == mkv_file_name:
            csv_mkv_pairs.append((csv_file, mkv_file))

# Split the video into samples using the timestamps
from tqdm import tqdm

for csv_file, mkv_file in tqdm(
    csv_mkv_pairs, desc="Splitting raw videos into samples", position=1, leave=False
):
    # Read the timestamps
    with open(csv_file) as csvfile:
        splitted_lines = [l.split(sep=",") for l in csvfile.readlines()]
    time_stamps = [
        (float(splits[0]) + float(splits[1]) / 30) for splits in splitted_lines
    ]
    # Split the video into samples
    for i, ts in tqdm(
        enumerate(time_stamps),
        desc=f"{mkv_file.split('/')[-1]}",
        position=0,
        leave=False,
    ):
        os.system(
            f"ffmpeg -hide_banner -loglevel error -y -ss {ts} -i {mkv_file} -frames:v 60 -vf scale=224:224 -c:v ffv1 ./dataset/train/{mkv_file.split('/')[-1].split('_')[0]}/{mkv_file.split('/')[-1].split('.')[0]}_{i+1}.avi"
        )
