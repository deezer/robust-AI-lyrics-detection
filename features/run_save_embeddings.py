import sys
from pathlib import Path

# Add project root to path for cross-module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from llm2vec import LLM2VecExtractor
from luar import Luar
from sbert import SentenceBert

from transcription.transcript_utils import LyricsProcessingOptions

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

LYRICS_OPTIONS_PERFECT = LyricsProcessingOptions(
    # POST
    normalize_lyrics=False,
    strip_lines=False,
    remove_thank_you=False,
    remove_emojis=False,
    no_speech_prob_threshold=0.0,
    remove_intro_outro=False,
    remove_only_punctuation=False,
    remove_subtitles=False,
    remove_be_right_back=False,
    remove_numbers_only=False,
    # PRE
    remove_verse_segmentation=True,
    remove_line_segmentation=False,
    remove_casing=False,
)

DO_PROCESS_LYRICS = True

DETECTORS = [
    "llm2vec",
    "sbert",
    "luar",
]

DATASET_PATHS = [""]

SAVE_SUFFIXES = [
    # "fwhisper-large-v3_DEFAULT_REAL",
    # "fwhisper-large-v3_DEFAULT_FAKE",
    # "fwhisper-large-v3_DEFAULT_FAKE-REAL",
    # "PERFECT-v3",
]

if DO_PROCESS_LYRICS:
    SAVE_SUFFIXES = [
        f"{suffix}_POST" if "PERFECT" not in suffix else f"{suffix}_PRE-remove_verses-v3"
        for suffix in SAVE_SUFFIXES
    ]

# XXX: adapt defaults here..
MODEL_LLM2VEC = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
# MODEL_LLM2VEC = "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp"
# MODEL_SBERT = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_SBERT = "BAAI/bge-m3"
LUAR_VARIANT = "CRUD"

datasets = {}
for DATASET_PATH, SAVE_SUFFIX in zip(DATASET_PATHS, SAVE_SUFFIXES):
    print(SAVE_SUFFIX)
    dataset = load_dataset(
        "robust_detection/detection/dataset_nlp4musa.py",
        "cls",
        data_dir=DATASET_PATH,
        trust_remote_code=True,
        lyrics_options=LYRICS_OPTIONS_PERFECT,
        do_process_lyrics=DO_PROCESS_LYRICS,
        use_transcripts=True if "perfect" not in SAVE_SUFFIX.lower() else False,
        cache_dir=None,
        download_mode="force_redownload",
    )
    datasets[SAVE_SUFFIX] = dataset


if __name__ == "__main__":
    for key, dataset in datasets.items():
        print(key)
        text_key = "transcription" if "real" in key.lower() or "fake" in key.lower() else "text"
        print(text_key)
        for detector in DETECTORS:
            if detector == "llm2vec":
                current_detector = LLM2VecExtractor(
                    current_train_subset="train",
                    current_subset="test",
                    dataset=dataset,
                    max_tokens=512,
                    _callername=detector,
                    protocol="nlp4musa",
                    model_llm2vec=MODEL_LLM2VEC,
                    model_llm2vec_short=MODEL_LLM2VEC.split("/")[-1].replace("-", "_"),
                    text_key=text_key,
                    embed_save_suffix=key,
                )
                current_detector.predict(do_save_and_classify=False, save_full_hf_dataset=True)
                print(f"Finished {key} with {detector}")
            elif detector == "sbert":
                current_detector = SentenceBert(
                    current_train_subset="train",
                    current_subset="test",
                    dataset=dataset,
                    max_tokens=512,
                    _callername=detector,
                    protocol="nlp4musa",
                    bert_variant=MODEL_SBERT,
                    text_key=text_key,
                    embed_save_suffix=key,
                )
                current_detector.predict(do_save_and_classify=False, save_full_hf_dataset=True)
                print(f"Finished {key} with {detector}")
            elif detector == "luar":
                current_detector = Luar(
                    current_train_subset="train",
                    current_subset="test",
                    dataset=dataset,
                    max_tokens=512,
                    _callername=detector,
                    protocol="nlp4musa",
                    luar_variant=LUAR_VARIANT,
                    text_key=text_key,
                    embed_save_suffix=key,
                )
                current_detector.predict(do_save_and_classify=False, save_full_hf_dataset=True)
                print(f"Finished {key} with {detector}")

    print(f"Done with all {len(datasets)} datasets")
