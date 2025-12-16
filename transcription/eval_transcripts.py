import argparse
import json
import os
from dataclasses import dataclass
from typing import Union

import numpy as np
from alt_eval import (  # NOTE: Requires alt_eval to be installed or available in PYTHONPATH
    compute_metrics,
)
from tqdm import tqdm
from transcript_utils import LYRICS_OPTIONS, LyricsNormalizer, process_transcripts

# Base directory - override via environment variable or command line
BASE_DIR = os.environ.get("PROJECT_BASE_DIR", ".")
DATA_FILENAMES = {
    "real": "real_songs.json",
    "fake": "fake_songs.json",
    "real_fake": "halffake_songs.json",
}


@dataclass
class Args:
    transcript_dir: str = "output/fwhisper-large-v2_DEFAULT_N5"
    filename: Union[str, None] = None
    mode: str = "real"
    postprocess_lyrics: bool = False


lyrics_normalizer = LyricsNormalizer(LYRICS_OPTIONS)


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transcript_dir",
        type=str,
        default="output/fwhisper-large-v2_DEFAULT_N5",
        help="Directory containing the transcripts.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Filename containing the transcripts (defaults to mode-specific file).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="real",
        help="Mode to evaluate: real, fake, real_fake",
    )
    parser.add_argument(
        "--postprocess_lyrics",
        action="store_true",
        help="Whether to postprocess the lyrics.",
    )


# Load JSON data from file
def load_json(file_path: str) -> dict:
    """Load JSON data from a given file path."""
    with open(file_path, "r") as f:
        return json.load(f)


def eval_transcripts(args, songs: dict) -> None:
    all_metrics = []
    for key in ["train", "test"]:
        for idx, song in enumerate(tqdm(songs[key])):
            song_title = song.get("LYRICS_TITLE", "Unknown_Title")
            artist_name = song.get("artist_name", "Unknown_Artist")
            song_language = song.get("language_str") or song.get("lang")
            song_genre = song.get("genre_name")
            song_main_genre = song.get("main_genre_name") or song.get("genre")
            song_model = song.get("model")

            md5 = song.get("md5")
            if not md5 and song["class"] != "generated":
                print(
                    f"Skipping song {song_title} by {artist_name} due to missing md5."
                )
                continue

            transcripts = song.get("transcription")

            if not transcripts:
                # print(
                #     f"Skipping song {song_title} by {artist_name} due to missing transcripts."
                # )
                continue
            if "fwhisper" in args.transcript_dir:
                try:
                    while "text" not in transcripts[0]:
                        transcripts = transcripts[0]
                        if (
                            transcripts == []
                            or all(x is None for x in transcripts)
                            or not transcripts
                        ):
                            transcripts = [{"text": " "}]
                            break
                except (IndexError, TypeError, KeyError):
                    continue
                transcripts = [transcripts]

            # GT
            gt = song.get("text")
            if gt is None:
                raise ValueError(
                    f"Missing ground truth lyrics for {song_title} by {artist_name}"
                )
            elif len(gt) == 0 or gt == " ":
                print(
                    f"SKIP: Empty ground truth lyrics for {song_title} by {artist_name}"
                )
                continue
            # alt-eval expects a list lyrics
            gt = [gt for _ in range(len(transcripts))]

            # TRANSCRIPTS
            all_processed_transcripts = process_transcripts(
                lyrics_normalizer,
                transcripts=transcripts,
                transcript_dir=args.transcript_dir,
                do_postprocess_lyrics=args.postprocess_lyrics,
                lyrics_normalizer_options=LYRICS_OPTIONS,
            )

            try:
                metrics = compute_metrics(
                    gt,
                    all_processed_transcripts,
                    languages=song_language if song_language else "en",
                )
            except Exception as e:
                print(f"Error computing metrics for {song_title}: {e}")
                continue

            all_metrics.append(
                {
                    "split": key,
                    "song_title": song_title,
                    "artist_name": artist_name,
                    "song_language": song_language,
                    "song_genre": song_genre,
                    "song_main_genre": song_main_genre,
                    "song_model": song_model,
                    "metrics": metrics,
                    "transcription_length": np.mean(
                        [len(transcript) for transcript in all_processed_transcripts]
                    ),
                    "gt_length": len(gt[0]),
                }
            )
            songs[key][idx]["metrics"] = {
                "metrics": metrics,
            }
    # save metrics
    save_str = os.path.join(
        args.transcript_dir, f"{DATA_FILENAMES[args.mode].split('.')[0]}"
    )
    if args.postprocess_lyrics:
        save_str += "_POST"
    with open(
        f"{save_str}_metrics-only.json",
        "w",
    ) as f:
        json.dump(all_metrics, f, indent=4)
    # save updated data
    with open(
        f"{save_str}_updated-all.json",
        "w",
    ) as f:
        json.dump(songs, f, indent=4)
    print(
        f"Saved metrics to {save_str}_metrics.json and updated data to {save_str}_metrics-ALL.json"
    )
    # save args
    with open(
        f"{save_str}_args.json",
        "w",
    ) as f:
        json.dump(LYRICS_OPTIONS.__dict__, f, indent=4)


def main(args: Args) -> None:
    if args.filename:
        transcript_filename = os.path.join(BASE_DIR, args.transcript_dir, args.filename)
    else:
        transcript_filename = os.path.join(
            BASE_DIR, args.transcript_dir, DATA_FILENAMES[args.mode]
        )
    print(f"Loading transcripts from {transcript_filename}.")
    try:
        songs = load_json(transcript_filename)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Failed to load transcripts from {transcript_filename}: {e}")
        return
    eval_transcripts(args, songs)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
