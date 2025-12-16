"""
Run feature for lyrics classification using SBERT embeddings.

Configuration parameters for lyrics processing and model settings.
Handles both real and AI-generated transcribed lyrics datasets.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path for cross-module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import DatasetDict, concatenate_datasets, load_dataset
from ensemble import Ensemble
from feature_extractor import FeatureExtractor
from gritlm import GritLMExtractor
from llm2vec import LLM2VecExtractor
from loglikelihood import LogLikelihood
from mms import MMS
from sbert import SentenceBert
from w2v2 import Wav2Vec2
from xeus import XEUS

from transcription.transcript_utils import LyricsProcessingOptions

NLL_MODELS = ["entropy", "max_nll", "perplexity", "mink_10"]


@dataclass
class Args:
    protocol: str = "emnlp"
    dataset_combo_name: str = ""
    real_dataset_name: str = "real_transcribed"
    fake_dataset_name: str = "fake_transcribed"
    only_real: bool = True
    real_dataset: str = "data/real/dataset.json"
    fake_dataset: str = "output/fwhisper-large-v2_DEFAULT/fake_songs.json"
    feature: str = "sbert"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    classifier: str = "mlp"
    n_neighbors: int = 3
    max_tokens: int = 512
    postprocess: bool = False
    strategy: Optional[str] = None  # for log-likelihood feature
    min_k: Optional[int] = None
    generation_idx: Optional[int] = None


@dataclass
class ModelForEnsemble:
    name: str
    feature_class: FeatureExtractor
    weight: int
    embedding_dim: Optional[int] = 768
    test_preds: Optional[List[List[List[float]]]] = None
    train_preds: Optional[List[List[List[float]]]] = None


def initialize_feature(args, unified_args, save_embeddings_dir):
    if args.feature == "sbert":
        feature = SentenceBert(
            model_name=args.model, max_tokens=args.max_tokens
        )
    elif args.feature == "loglikelihood":
        feature = LogLikelihood(
            model_name=args.model,
            strategy=args.strategy,
            min_k=args.min_k,
            max_tokens=args.max_tokens,
        )
    elif args.feature == "gritlm":
        feature = GritLMExtractor()
    elif args.feature == "llm2vec":
        feature = LLM2VecExtractor(
            max_tokens=args.max_tokens
        )
    elif args.feature == "xeus":
        feature = XEUS(
            model_name=args.model, max_tokens=args.max_tokens
        )
    elif args.feature == "mms":
        feature = MMS(
            model_name=args.model, max_tokens=args.max_tokens
        )
    elif args.feature == "w2v2":
        feature = Wav2Vec2(
            model_name=args.model, max_tokens=args.max_tokens
        )
    elif ["xeus", "mms", "w2v2"].count(args.feature) > 0:
        speech_args = args.detector_model.split("+")
        use_vad = "vad" in speech_args
        use_demucs = "demucs" in speech_args
        reencode = True if "128" in speech_args else False
        if args.detector == "mms":
            feature = MMS(
                model_variant=args.detector_model,
                use_vad=use_vad,
                use_demucs=use_demucs,
                reencode=reencode,
                **unified_args,
            )
        elif args.detector == "xeus":
            feature = XEUS(
                model_variant=args.detector_model,
                use_vad=use_vad,
                use_demucs=use_demucs,
                reencode=reencode,
                **unified_args,
            )
        elif args.detector == "w2v2":
            feature = Wav2Vec2(
                model_variant=args.detector_model,
                use_vad=use_vad,
                use_demucs=use_demucs,
                reencode=reencode,
                **unified_args,
            )
    else:
        raise ValueError(f"Unknown feature type: {args.feature}")

    feature.load()
    return feature


def get_save_dirs(args: Args) -> Tuple[str, str]:
    """
    Args:
        args (Args): Arguments

    Returns:
        Tuple[str, str]: Save directory for detector outputs and embeddings.
    Note:
        Embeddings are saved in a separate directory since they can be reused for different detector settings.
    """
    if ["llm2vec", "luar", "sbert", "binoculars", "gritlm"].count(args.detector) > 0:
        save_embeddings_dir = f"tok-{args.max_tokens}"
        if args.postprocess:
            save_embeddings_dir += "_POST"
        save_dir = f"{save_embeddings_dir}_{args.n_neighbors}-{args.classifier}"
    elif ["xeus", "mms", "w2v2"].count(args.detector) > 0:
        save_embeddings_dir = ""
        if args.postprocess:
            save_embeddings_dir += "_POST"
        save_dir = f"{args.detector_model}_{args.n_neighbors}-{args.classifier}"
    elif args.detector == "loglikelihood":
        save_embeddings_dir = ""
        if args.postprocess:
            save_embeddings_dir += "_POST"
        save_dir = f"{args.strategy}_{args.n_neighbors}-{args.classifier}"
    else:
        raise ValueError(f"Detector {args.detector} not supported.")

    save_dir = os.path.join(args.dataset_combo_name, save_dir)
    save_embeddings_dir = os.path.join(args.dataset_combo_name, save_embeddings_dir)
    return save_dir, save_embeddings_dir


def predict(args: Args, all_data=None, predict_only=False):
    lyrics_options = LyricsProcessingOptions(
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
        remove_verse_segmentation=False,
        remove_line_segmentation=False,
        remove_casing=False,
    )

    datasets = {
        args.real_dataset_name: args.real_dataset,
        args.fake_dataset_name: args.fake_dataset,
    }

    if args.only_real:
        datasets = {args.real_dataset_name: args.real_dataset}
        if (
            args.real_dataset_name != "real"
        ):  # XXX: ambiguous, actually it is just "non-transcribed"
            raise ValueError("ONLY_REAL is True, but real dataset name is not 'real'.")

    if all_data is None:
        dataset_dict = {}
        for i, (lyrics_set, lyrics_set_path) in enumerate(datasets.items()):
            dataset = load_dataset(
                f"robust_detection/dataset_{args.protocol}.py",
                "cls",
                data_dir=lyrics_set_path,
                trust_remote_code=True,
                lyrics_options=lyrics_options,
                do_process_lyrics=args.postprocess,
                generation_idx=args.generation_idx,
                split="test" if predict_only else None,
                use_transcripts=True if lyrics_set not in ["real", "fake"] else False,
                # download_mode="force_redownload"
                # verification_mode="no_checks",
            )
            dataset = dataset.map(lambda x: {"raw_label": lyrics_set})

            if not args.only_real:
                dataset = dataset.map(lambda x: {"label": i})
                if "real" in lyrics_set or "real_fake" in lyrics_set:
                    print(f"{lyrics_set}: filtering out generated lyrics.")
                    dataset = dataset.filter(lambda x: x["artist"] != "generated")
                elif "fake" in lyrics_set:
                    print(f"{lyrics_set}: filtering out real lyrics.")
                    dataset = dataset.filter(lambda x: x["artist"] == "generated")

            dataset_dict[lyrics_set] = dataset

        # CONCAT TO 1 DATASET
        if not predict_only:
            all_data_train = concatenate_datasets(
                [dataset["train"] for dataset in dataset_dict.values()]
            )
            all_data_test = concatenate_datasets(
                [dataset["test"] for dataset in dataset_dict.values()]
            )
            all_data = DatasetDict({"train": all_data_train, "test": all_data_test})
        else:
            all_data = concatenate_datasets(
                [dataset for dataset in dataset_dict.values()]
            )
            all_data = DatasetDict({"test": all_data})

    if "ensemble" not in args.detector:
        save_dir, save_embeddings_dir = get_save_dirs(args)

    # collect all non-model arguments here
    unified_args = {
        "dataset": all_data,
        "current_train_subset": "train" if not predict_only else None,
        "current_subset": "test",
        "protocol": args.protocol,
        "classifier": args.classifier,
        "n_neighbors": args.n_neighbors,
        "save_dir": save_dir if "ensemble" not in args.detector else None,
        "save_embeddings_dir": save_embeddings_dir
        if "ensemble" not in args.detector
        else None,
        "predict_only": predict_only,
    }
    # MULTIMODAL BRANCH
    if "ensemble" in args.detector:
        if "uniform" in args.detector:
            ensemble_weights = [1 / len(args.detector_model)] * len(args.detector_model)
        else:
            ensemble_weights = args.detector.split("_")[1].split(":")
        assert len(ensemble_weights) == len(args.detector_model)
        assert sum(float(w) for w in ensemble_weights) == 1.0

        ensemble_model_classes = []
        ensemble_save_dir = ""
        for model, weight in zip(args.detector_model, ensemble_weights):
            # some argument fiddling
            if ":" in model:
                detector_class, detector_model = model.split(":")
            else:
                detector_class = model
                detector_model = ""
            current_args = Args(**vars(args))
            current_args.detector_model = detector_model
            current_args.detector = detector_class

            if any([detector_class == nll for nll in NLL_MODELS]):
                current_args.strategy = detector_class
                current_args.detector = "loglikelihood"
            save_dir, save_embeddings_dir = get_save_dirs(current_args)
            unified_args["save_dir"] = save_dir
            unified_args["save_embeddings_dir"] = save_embeddings_dir
            detector = initialize_feature(
                current_args, unified_args, save_embeddings_dir
            )
            # build save_dir
            if any([detector_class == nll for nll in NLL_MODELS]):
                add_str = f"{current_args.strategy}-{detector.model_variant_short}"
            elif ["xeus", "mms", "w2v2"].count(detector_class) > 0:
                add_str = f"{detector.model_variant_short}-{detector_class}"
            else:
                add_str = (
                    f"tok-{current_args.max_tokens}-{detector.model_variant_short}"
                )
            ensemble_save_dir += str(weight) + add_str + "+"
            ensemble_model_classes.append(
                ModelForEnsemble(
                    name=detector_model, detector_class=detector, weight=float(weight)
                )
            )

        # gather unimodal predictions
        for model in ensemble_model_classes:
            (
                all_embeddings,
                labels,
                modelnames,
                artists,
                langs,
                genres,
                lyrics_id,
            ) = model.detector_class.predict(
                do_save_and_classify=False, save_full_hf_dataset=False
            )
            model.test_preds = all_embeddings["test"]
            if not predict_only:
                model.train_preds = all_embeddings["train"]
            model.embedding_dim = len(model.test_preds[0])

        ensemble_save_dir.strip("+")
        ensemble_save_dir += f"___{args.n_neighbors}-{args.classifier}"
        if args.postprocess:
            ensemble_save_dir += "_POST"
        ensemble_save_dir = os.path.join(args.dataset_combo_name, ensemble_save_dir)
        final_detector = Ensemble(
            **unified_args,
        )
        base_model_save_dir = os.path.join(
            final_detector.base_dir, final_detector.classifier, final_detector.protocol
        )
        model_save_dir = os.path.join(
            base_model_save_dir,
            args.dataset_combo_name,
            "ensemble",
            ensemble_save_dir,
        )
        combined_embs = final_detector.combine_embeddings(ensemble_model_classes)
        # embeddings already computed and potentially scaled, just train/eval final clf
        final_detector.classify_and_save(
            model_save_dir,
            combined_embs,
            labels,
            modelnames,
            artists,
            langs,
            genres,
            lyrics_id,
        )

        return model_save_dir, all_data

    # BACK TO UNIMODAL
    detector = initialize_feature(args, unified_args, save_embeddings_dir)

    model_save_dir = detector.predict(
        do_save_and_classify=True, save_full_hf_dataset=True
    )

    return model_save_dir, all_data


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Run detector")
    parser.add_argument(
        "--protocol",
        default=Args.protocol,
        choices=["nlp4musa", "emnlp"],
    )

    # Data arguments
    data_group = parser.add_argument_group("Data configuration")
    data_group.add_argument(
        "--real_dataset",
        default=Args.real_dataset,
        help="Path to real lyrics dataset",
    )
    data_group.add_argument(
        "--fake_dataset",
        default=Args.fake_dataset,
        help="Path to fake lyrics dataset",
    )
    data_group.add_argument(
        "--real_dataset_name",
        default=Args.real_dataset_name,
        help="Name for real dataset",
    )
    data_group.add_argument(
        "--fake_dataset_name",
        default=Args.fake_dataset_name,
        help="Name for fake dataset",
    )
    data_group.add_argument(
        "--only-real",
        action="store_true",
        default=Args.only_real,
        help="Use only real dataset to compare.",
    )
    data_group.add_argument(
        "--dataset_combo_name", default=Args.dataset_combo_name, help="Save suffix"
    )

    # Model arguments
    model_group = parser.add_argument_group("Model configuration")
    model_group.add_argument(
        "--detector",
        default=Args.detector,
        choices=["sbert", "luar", "llm2vec"],
        help="Detector type to use",
    )
    model_group.add_argument(
        "--detector_model",
        default=Args.detector_model,
        help="Model name",
    )
    model_group.add_argument(
        "--classifier",
        default=Args.classifier,
        help="Classifier to use, default: knn",
    )
    model_group.add_argument(
        "--n_neighbors",
        type=int,
        default=Args.n_neighbors,
        help="Number of neighbors for kNN classifier",
    )
    model_group.add_argument(
        "--max-tokens",
        type=int,
        default=Args.max_tokens,
        help="Maximum tokens for model input",
    )

    # Processing arguments
    proc_group = parser.add_argument_group("Lyrics processing")
    proc_group.add_argument(
        "--postprocess",
        action="store_true",
        default=Args.postprocess,
        help="Post-process transcriptions",
    )
    # TODO: pre-processing?
    args = parser.parse_args()

    return Args(**vars(args))


if __name__ == "__main__":
    args = parse_args()
    predict(args)
