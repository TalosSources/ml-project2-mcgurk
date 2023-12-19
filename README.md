# Investigating the McGurk effect on machine learning models

## Usage

### Requirements

- Download `ffpmeg` on your machine (used to process the raw dataset):

    - Windows 

    ```
    winget install --id=Gyan.FFmpeg  -e
    ```

    - macOS 

    ```
    brew install ffmpeg
    ```

    - Linux

    Using your favorite package manager!

- Install the Python requirements using `pip -r requirements.txt`
    
### Getting the dataset and processing it

- Run `setup.py`, which takes care of downloading the dataset and processing it to populate the dataset folder with the final training samples (in `dataset/train`).

### Running the Perceiver Mcgurk experiment
- Run the main.ipynb jupyter notebook
