import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import demucs.api
import torch
import torchaudio
from iso639 import Language
from transformers import (
    AutoProcessor,
    SeamlessM4TModel,
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


@dataclass
class Args:
    compute_type: str = "float16"
    device: str = "cuda"
    model: str = "facebook/mms-1b-all"
    whisper_model: str = "openai/whisper-large-v3"
    file_path: str = "data/real/dataset.json"
    output_dir: str = "output/mms_Wv3-LANG_N1"
    max_songs: int = 500_000
    n_runs: int = 1
    prefix: str | None = None
    provide_language: bool = False
    provide_whisper_language: bool = True
    vad_filter: bool = False
    use_demucs: bool = True
    demucs_dir: str = "demucs/real"


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
        "--model",
        type=str,
        default=Args.model,
        help="Model size (facebook/mms-1b-all, facebook/hf-seamless-m4t-large)",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default=Args.whisper_model,
        help="From OpenAI HF",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=Args.file_path,
        help="Path to the input JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=Args.output_dir,
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
        "--provide_language",
        action="store_true",
        default=Args.provide_language,
        help="Provide GT language of lyrics to the model",
    )
    parser.add_argument(
        "--provide_whisper_language",
        action="store_true",
        default=Args.provide_whisper_language,
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
        type=bool,
        default=Args.use_demucs,
        help="Use Demucs to separate audio before transcription",
    )
    parser.add_argument(
        "--demucs_dir",
        type=str,
        default=Args.demucs_dir,
        help="Directory to save Demucs separated audio",
    )
    return parser.parse_args()


# Load JSON data from file
def load_json(file_path: str) -> dict:
    """Load JSON data from a given file path."""
    with open(file_path, "r") as f:
        return json.load(f)


# Generate song path from md5 hash and quality
def py_song_path_from_md5(md5: str, quality: str = "mp3_128", base_path: str = None) -> str:
    """Generate the song path based on the md5 hash and quality."""
    if base_path is None:
        base_path = os.environ.get("AUDIO_BASE_PATH", "./data/audio")
    return "{}/{}/{}/{}/{}/{}.mp3".format(
        base_path, quality, md5[0], md5[1], md5[2], md5
    )


def separate_audio(song_path: str, song_md5: str, demucs_dir: str, device: str):
    """Separate the audio into vocals and accompaniment using Demucs."""
    out_path = f"{demucs_dir}/{song_md5}_vocals.wav"
    if os.path.exists(out_path):
        print(f"Vocals already separated for {song_md5}")
        return out_path
    separator = demucs.api.Separator(device=device)
    origin, separated = separator.separate_audio_file(song_path)
    vocals = separated["vocals"]
    if not os.path.exists(demucs_dir):
        os.makedirs(demucs_dir)
    with open(out_path, "wb") as f:
        demucs.api.save_audio(vocals, f.name, samplerate=separator.samplerate)
        print(f"Vocals separated for {song_md5}")
        return f.name


def transcribe_song(
    args: Args,
    song_path: str,
    model: SeamlessM4TModel,
    processor: AutoProcessor,
    song_language: Optional[str] = None,
):
    """Transcribe a song using faster-whisper."""
    results = []
    try:
        audio, orig_freq = torchaudio.load(song_path)
        audio = torchaudio.functional.resample(
            audio, orig_freq=orig_freq, new_freq=16_000
        )  # must be a 16 kHz waveform array
        # Get the number of samples in 10 seconds
        chunk_size = 16_000 * 30

        # Split the audio into chunks of 10 seconds
        chunks = audio.split(chunk_size, dim=-1)

        if args.model == "facebook/hf-seamless-m4t-large":
            if f"__{song_language}__" not in processor.feature_extractor.language_code:
                print(f"Language {song_language} not found, using English")
                song_language = "eng"
            audio_inputs = [
                processor(
                    audios=chunk,
                    return_tensors="pt",
                    src_lang=song_language,
                    sampling_rate=16_000,
                )
                for chunk in chunks
            ]
            audio_inputs = [audio_input.to(args.device) for audio_input in audio_inputs]
            if args.compute_type == "float16":
                audio_inputs = [
                    {k: v.half() for k, v in audio_input.items()}
                    for audio_input in audio_inputs
                ]
        else:
            try:
                processor.tokenizer.set_target_lang(song_language)
                model.load_adapter(song_language)
            except ValueError:
                print(f"Language {song_language} not found, using English")
                processor.tokenizer.set_target_lang("eng")
                model.load_adapter("eng")
            audio_inputs = [
                processor(
                    chunk.mean(dim=0, keepdim=False),
                    return_tensors="pt",
                    sampling_rate=16_000,
                )
                for chunk in chunks
            ]
            audio_inputs = [audio_input.to(args.device) for audio_input in audio_inputs]
            if args.compute_type == "float16":
                for idx, audio_input in enumerate(audio_inputs):
                    audio_inputs[idx]["input_values"] = audio_input[
                        "input_values"
                    ].half()
    except Exception as e:
        print(f"Error loading {song_path}: {e}")
        return [None]

    for i in range(args.n_runs):
        try:
            single_run_results = []
            for audio_input in audio_inputs:
                if args.model == "facebook/hf-seamless-m4t-large":
                    result = (
                        model.generate(
                            **audio_input, tgt_lang=song_language, generate_speech=False
                        )[0]
                        .cpu()
                        .numpy()
                        .squeeze()
                    )
                    # decode the result
                    single_run_result = processor.decode(result)[12:-5]
                else:
                    result = model(**audio_input, return_dict=True)[0]
                    ids = torch.argmax(result, dim=-1)[0]
                    single_run_result = processor.decode(ids)
                print(single_run_result)
                single_run_results.append(single_run_result)

            results.append(" ".join(single_run_results))
        except Exception as e:
            print(f"Error transcribing {song_path}: {e}")
            results.append(None)
    return results


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
    formatted_text = segments

    # Write the formatted transcription to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(formatted_text)

    print(f"Transcription saved to {file_path}")


def process_songs(
    songs: list,
    model: SeamlessM4TModel,
    processor: AutoProcessor,
    whisper_model: Optional[WhisperForConditionalGeneration],
    whisper_processor: Optional[WhisperProcessor],
    args: Args,
):
    """Loop through songs and transcribe each one if the file exists."""
    for key in ["train", "test"]:
        output_dir = os.path.join(args.output_dir, key)
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
                if args.provide_whisper_language:
                    audio, orig_freq = torchaudio.load(song_path)
                    audio = torchaudio.functional.resample(
                        audio, orig_freq=orig_freq, new_freq=16_000
                    )  # must be a 16 kHz waveform array
                    # remove channels
                    audio = audio.mean(dim=0, keepdim=False)
                    whisper_inputs = whisper_processor.feature_extractor(
                        audio, return_tensors="pt", sampling_rate=16_000
                    ).to(args.device)
                    if args.compute_type == "float16":
                        whisper_inputs = {
                            k: v.half() for k, v in whisper_inputs.items()
                        }
                    lang_token_id = whisper_model.generate(
                        **whisper_inputs, max_new_tokens=1
                    )[0, 1]
                    lang_token = whisper_processor.tokenizer.decode(lang_token_id)[2:4]
                    if lang_token == "jw":
                        lang_token = "jv"
                    song_language = Language.from_part1(lang_token).part3

                if args.use_demucs:
                    # Separate audio into vocals and accompaniment

                    song_path = separate_audio(
                        song_path, md5, args.demucs_dir, args.device
                    )
                # TRANSCRIBE!
                transcript = transcribe_song(
                    args,
                    song_path,
                    model,
                    processor=processor,
                    song_language=song_language,
                )

                # Save transcription to separate textfile for inspection, if available
                if transcript:
                    save_transcription(
                        song_title, artist_name, transcript, output_dir, idx
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
                with open(
                    os.path.join(args.output_dir, args.file_path.split("/")[-1]), "w"
                ) as f:
                    json.dump(songs, f, indent=4)
                break


def main(args: Args):  # Change function signature
    # Load dataset
    data = load_json(args.file_path)
    if args.provide_language and args.provide_whisper_language:
        raise ValueError(
            "Cannot provide both GT language and Whisper language to the model."
        )

    print("Loading SeamlessM4TModel models...")
    processor = AutoProcessor.from_pretrained(args.model)
    if args.model == "facebook/mms-1b-all":
        model = Wav2Vec2ForCTC.from_pretrained(args.model).to(args.device)
    elif args.model == "facebook/hf-seamless-m4t-large":
        model = SeamlessM4TModel.from_pretrained(args.model).to(args.device)
    else:
        raise ValueError(f"Model {args.model} not recognized")
    if args.compute_type == "float16":
        model.half().to(args.device)

    if args.provide_whisper_language:
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            args.whisper_model
        ).to(args.device)
        if args.compute_type == "float16":
            whisper_model.half().to(args.device)
        whisper_processor = WhisperProcessor.from_pretrained(args.whisper_model)

    process_songs(
        data,
        model,
        processor,
        whisper_model if args.provide_whisper_language else None,
        whisper_processor if args.provide_whisper_language else None,
        args,
    )


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
