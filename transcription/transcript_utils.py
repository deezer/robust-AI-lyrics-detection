import re
from dataclasses import dataclass
from typing import List

from alt_eval import normalize_lyrics


@dataclass
class LyricsProcessingOptions:
    # https://github.com/audioshake/alt-eval/blob/main/src/alt_eval/normalization.py
    normalize_lyrics: bool = True
    strip_lines: bool = True
    remove_thank_you: bool = True
    remove_emojis: bool = True
    # 1.0 means no filtering, but 0.9 worked best in my own experiments and in LyricWhiz
    # BUT: if using VAD, this should be 1.0 - the serve the same purpose.
    no_speech_prob_threshold: float = 0.9
    remove_intro_outro: bool = False
    remove_only_punctuation: bool = False
    remove_subtitles: bool = False
    remove_be_right_back: bool = False
    remove_numbers_only: bool = False
    # PREPROCESSING
    remove_verse_segmentation: bool = True
    remove_line_segmentation: bool = True
    remove_casing: bool = True


LYRICS_OPTIONS = LyricsProcessingOptions(
    normalize_lyrics=True,
    strip_lines=True,
    remove_thank_you=True,
    remove_emojis=True,
    no_speech_prob_threshold=0.9,
    remove_intro_outro=True,
    remove_only_punctuation=True,
    remove_subtitles=True,
    remove_be_right_back=True,
    remove_numbers_only=True,
)


class LyricsNormalizer:
    def __init__(self, options: LyricsProcessingOptions):
        self.options = options
        self.emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e6-\U0001f1ff"  # flags (subset of iOS flags)
            "]+",
            flags=re.UNICODE,
        )
        self.thank_you_pattern = re.compile(
            r"^\s*thank you[\s.,!?;:]*\s*$", re.IGNORECASE
        )
        # should handle "thanks!", "thanks.", "thanks", etc.
        self.thanks_pattern = re.compile(r"^\s*thanks[\s.,!?;:]*\s*$", re.IGNORECASE)
        # should handle "thanks for *.", "thanks for *!", "thanks for *", etc.
        self.thanks_for_pattern = re.compile(
            r"^\s*thanks for [\w\s.,!?;:]*\s*$", re.IGNORECASE
        )
        self.intro_outro_pattern = re.compile(
            r"^\s*(music\s+)?(intro|outro)[\s.,!?;:]*\s*$", re.IGNORECASE
        )
        self.punctuation_pattern = re.compile(r"(?<!\s)[.,!?;:]+(?!\s)")
        # Pattern for removing lines starting with subtitle indicators
        self.subtitle_pattern = re.compile(
            r"^\s*(Sous-titrage|Subtitle|Untertitel|Subtítulos|Legendas|Sottotitoli|字幕).*",
            re.IGNORECASE,
        )
        # remove "We'll be right back" lines
        self.be_right_back_pattern = re.compile(
            r"^\s*(we'll be right back|we will be right back)[\s.,!?;:]*\s*$",
            re.IGNORECASE,
        )
        # remove lines containing only numbers and whitespace
        self.remove_numbers_only_pattern = re.compile(r"^\s*\d+\s*$")

    def postprocess_lyrics(self, lyrics: List[str], speech_probs: List[float]) -> str:
        if self.options.strip_lines:
            lyrics = [line.strip() for line in lyrics]
        # XXX: this will not do anything for ALT metrics since it does Moses tokenization! But it matters for detectors.
        if self.options.remove_emojis:
            # remove emojis from the lyrics
            lyrics = [self.emoji_pattern.sub(r"", line) for line in lyrics]
        if self.options.remove_thank_you:
            # remove any form of "thank you" from the lyrics (e.g., "thank you", "thank you!", "Thank you.")
            lyrics = [self.thank_you_pattern.sub(r"", line) for line in lyrics]
            lyrics = [self.thanks_pattern.sub(r"", line) for line in lyrics]
            lyrics = [self.thanks_for_pattern.sub(r"", line) for line in lyrics]
        if self.options.remove_intro_outro:
            # remove lines that are only "intro" or "outro"
            lyrics = [self.intro_outro_pattern.sub(r"", line) for line in lyrics]
        if self.options.remove_only_punctuation:
            # remove lines that are only punctuation
            lyrics = [self.punctuation_pattern.sub(r"", line) for line in lyrics]
        if self.options.remove_subtitles:
            # remove lines starting with subtitle indicators
            lyrics = [self.subtitle_pattern.sub(r"", line) for line in lyrics]
        if self.options.remove_be_right_back:
            # remove "We'll be right back" lines
            lyrics = [self.be_right_back_pattern.sub(r"", line) for line in lyrics]
        if self.options.no_speech_prob_threshold:
            # remove lines with no_speech_prob < threshold
            lyrics = [
                line
                for line, prob in zip(lyrics, speech_probs)
                if prob <= self.options.no_speech_prob_threshold
            ]
        if self.options.remove_numbers_only:
            # remove lines containing only numbers and whitespace
            lyrics = [
                self.remove_numbers_only_pattern.sub(r"", line) for line in lyrics
            ]

        # strip empty lines
        lyrics = [line for line in lyrics if line]

        lyrics = "\n".join(lyrics)
        # this expects a single string
        if self.options.normalize_lyrics:
            lyrics = normalize_lyrics(lyrics)
        return lyrics

    def preprocess_lyrics(self, lyrics: str) -> str:
        if self.options.remove_verse_segmentation:
            # remove verse segmentation
            lyrics = lyrics.replace("\n\n", "\n")
        if self.options.remove_line_segmentation:
            # remove line segmentation
            lyrics = lyrics.replace("\n", " ")
        if self.options.remove_casing:
            # remove casing
            lyrics = lyrics.lower()
        return lyrics

    @staticmethod
    def transform_chirp_output(
        lyrics: List[str], confidences: List[float]
    ) -> List[dict]:
        # Chirp newlines are solely based on timestamp, there is no semantic meaning.
        # Instead, it splits lines/"short semantic units" by punctuation (". ")

        # but we need to keep the speech probabilities mapped to the correct "lines"
        # so we need to split the lyrics by ". " and keep the speech probs in sync
        processed_lyrics = []
        next_line_overlap = ""
        for line, prob in zip(lyrics, confidences):
            if not line:
                continue
            # split by ". " and keep the speech prob
            line_split = line.split(". ")
            for i, split_line in enumerate(line_split):
                if next_line_overlap:
                    processed_lyrics.append(
                        {
                            "text": (next_line_overlap + " " + split_line)
                            .strip(".")
                            .strip(),
                            "confidence": prob,
                        }
                    )
                if not line.endswith(".") and i == len(line_split) - 1:
                    # prepend to next line
                    next_line_overlap = split_line
                    continue
                else:
                    next_line_overlap = ""

                processed_lyrics.append(
                    {"text": split_line.strip(".").strip(), "confidence": prob}
                )
        return processed_lyrics

    @staticmethod
    def transform_seamless_output(lyrics: str) -> List[dict]:
        # comes in as simple string (not even newlines), should be formatted to other transcriptions
        # no splitting!
        return [{"text": lyrics, "no_speech_prob": 0.0}]


def process_transcripts(
    lyrics_normalizer,
    transcripts,
    transcript_dir,
    do_postprocess_lyrics: bool = True,
    lyrics_normalizer_options: LyricsProcessingOptions = LYRICS_OPTIONS,
):
    lyrics_normalizer = LyricsNormalizer(lyrics_normalizer_options)

    confidence_str = "confidence" if "chirp" in transcript_dir else "no_speech_prob"
    if (
        isinstance(transcripts, list)
        and isinstance(transcripts[0], list)
        and len(transcripts) == 2
    ):
        # flatten each suno-generated transcript
        # XXX: this treats different suno generations and their transcripts the same!

        transcripts = [generation for sublist in transcripts for generation in sublist]
    all_processed_transcripts = []
    for transcript in transcripts:
        if transcript is None:
            all_processed_transcripts.append("")
            continue
        if "chirp" in transcript_dir:
            transcript = lyrics_normalizer.transform_chirp_output(
                [line["text"] for line in transcript],
                [line[confidence_str] for line in transcript],
            )
        elif "sM4T" in transcript_dir or "mms" in transcript_dir:
            transcript = lyrics_normalizer.transform_seamless_output(transcript)
        if do_postprocess_lyrics:
            processed_transcript = lyrics_normalizer.postprocess_lyrics(
                [line["text"] for line in transcript],
                [line[confidence_str] for line in transcript],
            )
        else:
            processed_transcript = "\n".join(
                [line["text"].strip() for line in transcript]
            )
        all_processed_transcripts.append(processed_transcript)
    return all_processed_transcripts
