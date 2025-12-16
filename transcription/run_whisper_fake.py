import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from adversarial_augmenter import AdversarialAugmenter
from faster_whisper import WhisperModel
from pedalboard.io import AudioFile
from silero_vad import read_audio
from utils import build_output_dir, download_song_suno, load_json, separate_audio
from utils import transcribe_song_whisper as transcribe_song

# def _default_augmentation_list():
#     return ["reverb"]


def get_default_file_path(mode: str, model: str) -> str:
    if model == "udio":
        return "data/generated/udio/dataset.json"
    if mode == "original":
        return "data/generated/suno/halffake_songs.json"
    elif mode == "fake":
        return "data/generated/suno/fake_songs.json"


@dataclass
class Args:
    compute_type: str = "float16"
    device: str = "cuda"
    beam_size: int = 5
    model: str = "large-v2"
    file_path: Union[str, None] = None
    output_base_dir: str = "output_MODE/fwhisper"
    vad_filter: bool = False
    max_songs: int = 500_000
    n_runs: int = 1
    prefix: str | None = None  # Use str | None for optional string arguments
    provide_language: bool = False
    mode: str = "fake"  # original/fake
    use_demucs: bool = False
    continue_from_existing_transcriptions: bool = False
    augmentation_list: Union[str, None] = None
    generation_model: str = "suno"


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe lyrics from audio files.")
    parser.add_argument(
        "--compute_type",
        type=str,
        default=Args.compute_type,
        help="Compute type (float16 or int8)",
    )
    parser.add_argument(
        "--device", type=str, default=Args.device, help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--beam_size", type=int, default=Args.beam_size, help="Beam size for decoding"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=Args.model,
        help="Model size (large-v2/large-v3/deepdml/faster-whisper-large-v3-turbo-ct2)",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=Args.file_path,
        help="Path to the input JSON file",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=Args.output_base_dir,
        help="Directory to save transcriptions",
    )
    parser.add_argument(
        "--max_songs",
        type=int,
        default=Args.max_songs,
        help="Maximum number of songs to process",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=Args.n_runs,
        help="Number of Whisper runs per song",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=Args.prefix,
        help="Prefix for the transcription prompt (e.g., 'lyrics:')",
    )
    parser.add_argument(
        "--provide_language",
        action="store_true",
        default=Args.provide_language,
        help="Provide GT language of lyrics to the model",
    )
    parser.add_argument(
        "--vad_filter",
        action="store_true",
        default=Args.vad_filter,
        help="Apply voice activity detection filter",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=Args.mode,
        help="Mode to run in (original/fake)",
    )
    parser.add_argument(
        "--use_demucs",
        action="store_true",
        default=Args.use_demucs,
        help="Use Demucs to separate audio before transcription",
    )
    parser.add_argument(
        "--continue_from_existing_transcriptions",
        "-c",
        action="store_true",
        default=Args.continue_from_existing_transcriptions,
        help="Continue from existing transcriptions",
    )
    parser.add_argument(
        "--augmentation_list",
        type=str,
        nargs="+",
        default=Args.augmentation_list,
        help="Augmentation list to use",
    )
    parser.add_argument(
        "--generation_model",
        type=str,
        default=Args.generation_model,
        help="Model used for generation (suno/lyrics)",
    )

    return parser.parse_args()


def save_transcription(
    song_genre: str,
    song_language: str,
    song_model: str,
    transcription: dict,
    output_dir: str,
    song_number: int,
    suno_idx: int,
):
    """Save the transcription to a text file, formatted with segments separated by newlines."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate a safe file name by replacing spaces and special characters
    file_name = f"{song_number:04d}_{suno_idx}_{song_genre}_{song_language}_{song_model}.txt".replace(
        " ", "_"
    ).replace(
        "/", "_"
    )
    file_path = os.path.join(output_dir, file_name)

    # Format the transcription
    formatted_text = ""
    if all(t is None for t in transcription):
        print(f"Transcription failed for {song_genre} - {song_language} - {song_model}")
        return
    segments = transcription[0]  # XXX: this only takes the first run
    for segment in segments:
        formatted_text += f"{segment['text']}\n"

    # Write the formatted transcription to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(formatted_text)

    print(f"Transcription saved to {file_path}")


def process_songs(songs: list, model, args: Args):
    """Loop through songs and transcribe each one if the file exists."""
    for key in ["train", "test"]:
        output_dir_printing = os.path.join(args.output_dir, key)
        for idx, song in enumerate(songs[key]):
            if idx >= args.max_songs:
                break
            if song["class"] != "generated" and args.mode == "fake":
                continue
            elif song["class"] != "original" and args.mode == "original":
                continue

            song_genre = song.get("label_genre")
            language = song.get("label_lang")
            song_model = song.get("label_model")
            mp3_urls = song.get("mp3_urls")
            if mp3_urls is None or all([url is None for url in mp3_urls]):
                print(
                    f"No mp3 URLs available for {song_genre} - {language} - {song_model}"
                )
                continue

            if "transcription" in song and "song_paths" in song:
                print(
                    f"Skipping song {idx + 1}/{len(songs[key])}: {song_genre} - {language} - {song_model}, already processed."
                )
                continue

            # udio files are stored locally
            if args.generation_model == "udio":
                song_paths = mp3_urls
            else:
                song_paths = download_song_suno(mp3_urls, args.output_dir)

            all_transcripts = []
            for suno_idx, song_path in enumerate(song_paths):
                if os.path.exists(song_path):
                    print(
                        f"Processing song {idx}/{len(songs[key])}: {song_genre} - {language} - {song_model}"
                    )

                    song_language = (
                        song.get("language_str") if args.provide_language else None
                    )
                    suno_id = song_path.split("/")[-1].split(".")[0]
                    if args.augmentation_list:
                        save_dir = (
                            Path(args.output_base_dir).parents[1]
                            / f"output_augmented/{args.mode}/{'_'.join(args.augmentation_list)}"
                        )
                        augmented_song_path = str(save_dir) + f"/{suno_id}.wav"
                        if not os.path.exists(augmented_song_path):
                            os.makedirs(save_dir, exist_ok=True)
                            audio = read_audio(
                                song_path
                            )  # auto-converts to mono, 16kHz
                            augmenter = AdversarialAugmenter(args.augmentation_list)
                            augmented_audio = augmenter.py_apply_augmentation(audio)

                            with AudioFile(
                                augmented_song_path,
                                "w",
                                samplerate=augmenter.sample_rate,
                                num_channels=1,
                            ) as f:
                                f.write(augmented_audio)
                        song_path = augmented_song_path
                    if args.use_demucs:
                        # Separate audio into vocals and accompaniment
                        demucs_dir = Path(args.output_base_dir) / "demucs" / args.mode
                        if args.augmentation_list:
                            demucs_dir = demucs_dir / "_".join(args.augmentation_list)
                        song_path = separate_audio(
                            song_path, suno_id, demucs_dir, args.device
                        )
                    # TRANSCRIBE!
                    transcript = transcribe_song(
                        args, song_path, model, song_language=song_language
                    )
                    # Save transcription to separate textfile for inspection, if available
                    if transcript:
                        save_transcription(
                            song_genre,
                            language,
                            song_model,
                            transcript,
                            output_dir_printing,
                            idx,
                            suno_idx,
                        )
                    all_transcripts.append(transcript)
                else:
                    print(
                        f"Song file does not exist for {song_genre} - {language} - {song_model}"
                    )
            print("-" * 80)
            songs[key][idx]["transcription"] = all_transcripts
            songs[key][idx]["song_paths"] = song_paths

            # save updated data
            try:
                with open(
                    os.path.join(args.output_dir, args.file_path.split("/")[-1]), "w"
                ) as f:
                    json.dump(songs, f, indent=4)
            except KeyboardInterrupt:
                print("Interrupted, saving data")
                with open(
                    os.path.join(args.output_dir, args.file_path.split("/")[-1]), "w"
                ) as f:
                    json.dump(songs, f, indent=4)
                break
            # save args
            with open(os.path.join(args.output_dir, "args.json"), "w") as f:
                json.dump(args.__dict__, f, indent=4)


def main(args: Args):
    if args.generation_model == "udio":
        args.output_base_dir = args.output_base_dir.replace("MODE", "MODE_udio")
    if args.mode == "original":
        args.output_base_dir = args.output_base_dir.replace("MODE", "real_fake")
    elif args.mode == "fake":
        args.output_base_dir = args.output_base_dir.replace("MODE", "fake")

    else:
        raise ValueError("Invalid mode")
    args.output_dir = build_output_dir(args)

    if args.generation_model not in ["suno", "udio"]:
        raise ValueError("Invalid generation model.")
    if args.generation_model == "udio" and args.mode == "original":
        raise ValueError("Cannot use udio generation model for original data.")

    if not args.file_path:
        args.file_path = get_default_file_path(args.mode, args.generation_model)

    if args.continue_from_existing_transcriptions:
        file_path = os.path.join(args.output_dir, args.file_path.split("/")[-1])
    else:
        file_path = args.file_path
    print(file_path)
    data = load_json(file_path)

    print("Loading faster-whisper models...")
    if args.model == "large-v3-turbo":
        model_name = "deepdml/faster-whisper-large-v3-turbo-ct2"
    else:
        model_name = args.model
    model = WhisperModel(
        model_name,
        compute_type=args.compute_type,
        device=args.device,
    )

    process_songs(data, model, args)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
