import json
import os
import pathlib
from typing import List, Optional
from urllib.request import urlretrieve

import demucs.api


def separate_audio(song_path: str, song_md5: str, demucs_dir: str, device: str):
    """Separate the audio into vocals and accompaniment using Demucs."""
    out_path = f"{demucs_dir}/{song_md5}_vocals.wav".replace("|", "_")
    if os.path.exists(out_path):
        # print(f"Vocals already separated for {song_md5}")
        return out_path
    separator = demucs.api.Separator(device=device)
    origin, separated = separator.separate_audio_file(song_path)
    vocals = separated["vocals"]
    if not os.path.exists(demucs_dir):
        os.makedirs(demucs_dir)
    with open(out_path, "wb") as f:
        demucs.api.save_audio(vocals, f.name, samplerate=separator.samplerate)
        # print(f"Vocals separated for {song_md5}")
        return f.name


# Load JSON data from file
def load_json(file_path: str) -> dict:
    """Load JSON data from a given file path."""
    with open(file_path, "r") as f:
        return json.load(f)


# Generate song path from md5 hash and quality
def py_song_path_from_md5(md5: str, quality: str = "mp3_128", base_path: str = None) -> str:
    """Generate the song path based on the md5 hash and quality.

    Args:
        md5: Song MD5 hash
        quality: Audio quality folder name
        base_path: Base path for audio storage. Defaults to AUDIO_BASE_PATH env var or ./data/audio
    """
    if base_path is None:
        base_path = os.environ.get("AUDIO_BASE_PATH", "./data/audio")
    return "{}/{}/{}/{}/{}/{}.mp3".format(
        base_path, quality, md5[0], md5[1], md5[2], md5
    )


def download_song_suno(mp3_urls: List[str], output_dir) -> List[str]:
    """Download songs from Suno.

    Args:
        mp3_urls (List[str]): Web URLs to the mp3 files.

    Returns:
        List[str]: Local paths to the downloaded mp3 files.
    """
    suno_base_url = "https://cdn1.suno.ai/"
    base_path = str(pathlib.Path(output_dir).parent) + "/mp3"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    song_paths = []
    for mp3_url in mp3_urls:
        # Download the mp3 file
        song_id = (
            mp3_url.split("/")[-1].split(".mp3")[0].split("?item_id=")[-1] + ".mp3"
        )
        song_path = os.path.join(base_path, song_id)
        if not os.path.exists(song_path):
            suno_song_url = suno_base_url + song_id
            print(f"Downloading {suno_song_url} to {song_path}")
            urlretrieve(suno_song_url, song_path)
        song_paths.append(song_path)
    return song_paths


def transcribe_song_whisper(
    args,
    song_path: str,
    model,
    song_language: Optional[str] = None,
):
    """Transcribe a song using faster-whisper."""
    results = []
    for i in range(args.n_runs):
        try:
            result, _ = model.transcribe(
                song_path,
                language=song_language,
                beam_size=args.beam_size,
                word_timestamps=True,
                vad_filter=args.vad_filter,
                initial_prompt=args.prefix,
            )
            result = list(result)
            print([segment.text for segment in result])

            result = [
                {
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "no_speech_prob": segment.no_speech_prob,
                    "words": [word._asdict() for word in segment.words],
                }
                for segment in result
            ]
            results.append(result)
        except Exception as e:
            print(f"Error transcribing {song_path}: {e}")
            results.append(None)
    return results


def build_output_dir(args) -> str:
    """Build the output directory for the transcriptions."""
    output_dir = args.output_base_dir
    if args.model == "deepdml/faster-whisper-large-v3-turbo-ct2":
        model_name = "large-v3-turbo"
    else:
        model_name = args.model
    output_dir = output_dir + "-" + model_name
    if args.vad_filter:
        output_dir += "_VAD"
    if args.prefix == "lyrics:":
        output_dir += "+PREFIX"
    elif args.prefix:
        output_dir += f"+PREFIX-{args.prefix}"
    if args.provide_language:
        output_dir += "+LANG"
    if args.use_demucs:
        output_dir += "_DEMUCS"
    if args.augmentation_list:
        output_dir += "_" + "+".join(args.augmentation_list)
    if (
        not args.vad_filter
        and not args.prefix
        and not args.provide_language
        and not args.use_demucs
        and not args.augmentation_list
    ):
        output_dir += "_DEFAULT"
    output_dir += f"_N{args.n_runs}"
    return output_dir
