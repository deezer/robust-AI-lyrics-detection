import os

import tqdm
from eval_transcripts import Args, main

# Base directory - override via environment variable
BASE_DIR = os.environ.get("PROJECT_BASE_DIR", ".")

DIRS = {
    "output": "real",
    "output_fake": "fake",
    "output_real_fake": "real_fake",
}
DATA_FILENAMES = {
    "real": "real_songs.json",
    "fake": "fake_songs.json",
    "real_fake": "halffake_songs.json",
}

for dir_name, mode in DIRS.items():
    current_dir = os.path.join(BASE_DIR, dir_name)
    # listdir
    for transcriber in tqdm.tqdm(os.listdir(current_dir)):
        # if "fwhisper" not in transcriber:
        #     continue
        for filename in os.listdir(os.path.join(current_dir, transcriber)):
            # take from DATA_FILENAMES
            data_filename = DATA_FILENAMES[mode]
            if filename == data_filename:
                print(
                    f"Found {data_filename} in {os.path.join(current_dir, transcriber)}"
                )
                break
        else:
            print(
                f"Did not find {data_filename} in {os.path.join(current_dir, transcriber)}"
            )
            continue
        if not os.path.exists(os.path.join(current_dir, transcriber, filename)):
            print(f"Skipping {os.path.join(current_dir, transcriber, filename)}")
            continue
        if "sM4" not in os.path.join(current_dir, transcriber):
            continue
        print(os.path.join(current_dir, transcriber))
        args = Args(
            transcript_dir=os.path.join(current_dir, transcriber),
            filename=data_filename,
            mode=mode,
        )
        main(args)
