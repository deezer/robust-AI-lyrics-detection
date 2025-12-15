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
from utils import build_output_dir, load_json, py_song_path_from_md5, separate_audio
from utils import transcribe_song_whisper as transcribe_song


@dataclass
class Args:
    compute_type: str = "float16"
    device: str = "cuda"
    beam_size: int = 5
    model: str = "large-v3"
    file_path: str = "data/real/dataset.json"
    output_base_dir: str = "output/fwhisper"
    max_songs: int = 500_000
    n_runs: int = 1
    # lyrics: prompt taken from LYRICWHIZ: ROBUST MULTILINGUAL ZERO-SHOT LYRICS TRANSCRIPTION BY WHISPERING TO CHATGPT (https://arxiv.org/abs/2306.17103)
    # XXX: they translate the prompt to the target language, but we can't do that here (e.g., "paroles:" in French)
    prefix: str | None = None
    provide_language: bool = False
    vad_filter: bool = False
    use_demucs: bool = False
    continue_from_existing_transcriptions: bool = False
    augmentation_list: Union[str, None] = None
    generation_model: Union[str, None] = None


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
        help="Model size (large-v2/large-v3/large-v3-turbo)",
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
        help="Augmentation list to use (only one in practice); augmentation=atack=perturbation",
    )
    parser.add_argument(
        "--generation_model",
        type=str,
        default=Args.generation_model,
        help="Model used for generation (suno/lyrics)",
    )
    return parser.parse_args()


def save_transcription(
    song_title: str,
    artist_name: str,
    transcription: dict,
    output_dir: str,
    song_number: int,
):
    """Save the transcription to a text file, formatted with segments separated by newlines."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate a safe file name by replacing spaces and special characters
    file_name = f"{song_number:04d}_{song_title}_{artist_name}.txt".replace(
        " ", "_"
    ).replace("/", "_")
    file_path = os.path.join(output_dir, file_name)

    # Format the transcription
    formatted_text = ""
    if all(t is None for t in transcription):
        print(f"Transcription failed for {song_title} by {artist_name}")
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
            if "merged_md5" in song:
                continue
            md5 = song.get("md5")
            song_title = song.get("LYRICS_TITLE", "Unknown_Title")
            artist_name = song.get("artist_name", "Unknown_Artist")

            if "transcription" in song:
                print(
                    f"Skipping song {song_title} by {artist_name} due to existing transcription."
                )
                continue

            if not md5:
                print(
                    f"Skipping song {song_title} by {artist_name} due to missing md5."
                )
                continue

            song_path = py_song_path_from_md5(md5, quality="mp3_128")

            if os.path.exists(song_path):
                print(
                    f"Processing song {idx}/{len(songs[key])}: {song_title} by {artist_name}"
                )

                song_language = (
                    song.get("language_str") if args.provide_language else None
                )
                if args.augmentation_list:
                    save_dir = (
                        Path(args.output_base_dir).parents[1]
                        / f"output_augmented/{'_'.join(args.augmentation_list)}"
                    )
                    augmented_song_path = str(save_dir) + f"/{md5}.wav"
                    if not os.path.exists(augmented_song_path):
                        os.makedirs(save_dir, exist_ok=True)
                        audio = read_audio(song_path)  # auto-converts to mono, 16kHz
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
                    demucs_dir = Path(args.output_base_dir) / "demucs" / "real"
                    if args.augmentation_list:
                        demucs_dir = demucs_dir / "_".join(args.augmentation_list)
                    song_path = separate_audio(song_path, md5, demucs_dir, args.device)
                # TRANSCRIBE!
                transcript = transcribe_song(
                    args, song_path, model, song_language=song_language
                )

                # Save transcription to separate textfile for inspection, if available
                if transcript:
                    save_transcription(
                        song_title, artist_name, transcript, output_dir_printing, idx
                    )
            else:
                print(f"Song file does not exist for {song_title} by {artist_name}")
            print("-" * 80)
            songs[key][idx]["transcription"] = transcript

            # save updated data
            try:
                with open(
                    os.path.join(args.output_dir, args.file_path.split("/")[-1]), "w"
                ) as f:
                    json.dump(songs, f, indent=4)
            except KeyboardInterrupt:
                print("Interrupted, saving data")
                with open(os.path.join(args.args.file_path.split("/")[-1]), "w") as f:
                    json.dump(songs, f, indent=4)
                break
            # save args
            with open(os.path.join(args.output_dir, "args.json"), "w") as f:
                json.dump(args.__dict__, f, indent=4)


def main(args: Args):
    args.output_dir = build_output_dir(args)
    if args.generation_model == "udio":
        args.file_path = "data/generated/udio/dataset.json"
        args.output_dir = args.output_dir.replace("output", "output_udio")
    elif args.generation_model:
        raise ValueError(f"Invalid generation model: {args.generation_model}")

    if args.continue_from_existing_transcriptions:
        file_path = os.path.join(args.output_dir, args.file_path.split("/")[-1])
    else:
        file_path = args.file_path
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
