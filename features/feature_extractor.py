import copy
import json
import os

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import joblib
import lightning.pytorch as pl
import numpy as np
import torch
from lit_mlp import LitMLP, TwoStreamLitMLP
from pedalboard.io import AudioFile
from pydub import AudioSegment
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from transcription.adversarial_augmenter import AdversarialAugmenter
from transcription.utils import separate_audio

AUDIO_ATTACKS = ["reverb", "pitch", "eq", "noise", "stretch"]


class FeatureExtractor(ABC):
    def __init__(
        self,
        dataset,
        save_dir: Optional[str] = None,
        save_embeddings_dir: Optional[str] = None,
        max_tokens: Optional[int] = 512,
        hf_token: Optional[str] = None,  # better set via hf cli login
        classifier: Optional[
            str
        ] = None,  # mlp, knn, lit_mlp (use last for runs in paper/best performance)
        protocol: Optional[str] = None,  # now only use NLP4MUSA
        current_train_subset: Union[str, None] = None,
        current_subset: str = "test",
        n_neighbors: int = 3,
        # NLL args
        logprob_dir: Optional[str] = None,
        logprob_model: Optional[str] = None,
        strategy: Optional[str] = None,
        # speech stuff
        use_demucs: bool = False,
        use_vad: bool = False,
        # skip training, only predict with loaded classifier
        predict_only: bool = False,
        # reencode audio to 128 kb/s if not already
        reencode: bool = False,
    ):
        self.save_dir = save_dir
        self.save_embeddings_dir = save_embeddings_dir
        self.n_neighbors = n_neighbors
        self.CURRENT_SUBSET = current_subset
        self.CURRENT_TRAIN_SUBSET = current_train_subset
        self.subsets = [current_subset, current_train_subset]
        self.subsets = [s for s in self.subsets if s]
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.hf_token = hf_token
        self.classifier = classifier
        self.protocol = protocol
        self.base_dir = os.environ.get("ARTEFACTS_DIR", "./artefacts")
        self.logprob_dir = logprob_dir
        self.logprob_model = logprob_model
        self.strategy = strategy
        self.use_demucs = use_demucs
        self.use_vad = use_vad
        self.scalers = None
        self.predict_only = predict_only
        self.reencode = reencode

        if self.classifier not in ["knn", "mlp", "lit_mlp"]:
            raise ValueError(f"The classifier type {self.classifier} is not supported!")

    def _get_model_variant_short(self):
        return self.model_variant.split("/")[-1] if self.model_variant else "base"

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def compute_embeddings(self):
        pass

    def predict(
        self, do_save_and_classify: bool = True, save_full_hf_dataset: bool = False
    ):
        # only load model if required
        model = None
        if self.use_vad:
            vad = load_silero_vad()

        # to keep track of metadata
        lyrics_id = {s: [] for s in self.subsets}
        embs = {s: [] for s in self.subsets}
        artists = {s: [] for s in self.subsets}
        langs = {s: [] for s in self.subsets}
        genres = {s: [] for s in self.subsets}
        labels = {s: [] for s in self.subsets}
        modelnames = {s: [] for s in self.subsets}

        # Load the vectors if existing
        if self.model_type == "loglikelihood":
            model_type_str = f"{self.model_type}-{self.strategy}"
        else:
            model_type_str = self.model_type

        # path to feature-specific embedding vectors
        path_vectors = f"{self.base_dir}/vectors/{model_type_str}/{self.save_embeddings_dir}_{self.model_variant_short}.json"
        print(path_vectors)
        if os.path.isfile(path_vectors):
            print(f"Loading vectors from {path_vectors}")
            f_vectors = open(path_vectors, "r")
            all_vectors = json.load(f_vectors)
            f_vectors.close()
        else:
            all_vectors = {"train": {}, "test": {}}

        updated_elements_cpt = 0

        # dataset is used to make analyses of embeddings
        dataset = copy.deepcopy(self.dataset)
        for subset in self.subsets:
            dataset[subset] = dataset[subset].add_column(
                "embeddings", [[] for _ in range(len(dataset[subset]))]
            )
        for effect in AUDIO_ATTACKS:
            if effect in self.save_embeddings_dir:
                # XXX: we only consider one attack at a time.
                attack_to_apply = effect
                break
        else:
            attack_to_apply = None

        for subset in self.subsets:
            vectors_subset = "train" if "train" in subset else "test"
            if self.predict_only and vectors_subset == "train":
                print("Skipping training set for prediction only")
                continue

            # Load logprobs if existing
            # used for NLL-based (probabilistic) feature features
            if self.logprob_dir:
                path_logprobs = f"{self.base_dir}/log_probs/{self.save_embeddings_dir}_{self.model_variant_short}_{subset}.json"
                print(path_logprobs)
                if os.path.isfile(path_logprobs):
                    with open(path_logprobs, "r") as f_logprobs:
                        logprobs = json.load(f_logprobs)
                else:
                    raise FileNotFoundError(
                        f"Log probability file not found: {path_logprobs}\n"
                        f"Pre-compute log probs using an LLM and save to this path, "
                        f"or set logprob_dir=None to skip NLL-based features."
                    )

                if len(logprobs) != len(dataset[subset]):
                    raise ValueError(
                        f"Length mismatch between logprobs ({len(logprobs)}) and "
                        f"dataset ({len(dataset[subset])}) for {subset}.\n"
                        f"Re-compute log probs for: {path_logprobs}"
                    )

            for idx, d in enumerate(tqdm(dataset[subset])):
                lyrics = d["lyrics"]
                mp3_path = d["mp3_paths"]

                # empty original lyrics are already filtered out in the dataset.
                # if they are empty here, it is due to poor transcripts, and we don't want to filter for that.
                if len(lyrics) == 0:
                    lyrics = " "

                lyrics_id[subset].append(d["id"])
                artists[subset].append(d["artist"])
                langs[subset].append(d["lang"])
                genres[subset].append(d["genre"])
                labels[subset].append(d["label"])
                modelnames[subset].append(d["model_name"])

                # stricter matching: both id and raw_label must match
                vector_key = f"{d['id']}_{d['raw_label']}"
                if vector_key in all_vectors[vectors_subset]:
                    stored_data = all_vectors[vectors_subset][vector_key]
                    embs[subset].append(stored_data["vectors"])
                    continue

                # for speed-up, only load model when needed
                if not model:
                    model = self.load_model()

                # SPEECH PROCESSING
                if self.reencode and d["label"] == 1:
                    # needed since GT songs are 128kbps and AI-generated ones 192kbps, which could introduce bias
                    new_mp3_path = mp3_path.replace("output", "output/128kbps")
                    if not os.path.exists(os.path.dirname(new_mp3_path)):
                        os.makedirs(os.path.dirname(new_mp3_path), exist_ok=True)
                    # reencode if not already
                    if os.path.exists(new_mp3_path):
                        mp3_path = new_mp3_path
                    else:
                        audio = AudioSegment.from_file(mp3_path)
                        audio.export(
                            new_mp3_path,
                            format="mp3",
                            bitrate="128k",
                            codec="libmp3lame",
                        )
                        mp3_path = new_mp3_path

                # TODO: move this to a separate function
                if attack_to_apply and (self.use_demucs or self.use_vad):
                    # once here, we change the mp3 path to be loaded
                    if not mp3_path:
                        raise ValueError(f"{d['id']} has no mp3 paths! {d}")
                    if "fake" in d["raw_label"]:
                        aa_dir = "output_aa/fake"
                    elif "real" in d["raw_label"]:
                        aa_dir = "output_aa/real"
                    aa_dir = os.path.join(aa_dir, attack_to_apply)
                    if not os.path.exists(aa_dir):
                        os.makedirs(aa_dir, exist_ok=True)
                    aa_mp3_path = os.path.join(
                        aa_dir,
                        mp3_path.replace(".mp3", f"_{attack_to_apply}.mp3").split("/")[
                            -1
                        ],
                    )
                    if not os.path.exists(aa_mp3_path):
                        audio = read_audio(mp3_path)  # auto-converts to mono, 16kHz
                        augmenter = AdversarialAugmenter([attack_to_apply])
                        augmented_audio = augmenter.py_apply_augmentation(audio)

                        with AudioFile(
                            aa_mp3_path,
                            "w",
                            samplerate=augmenter.sample_rate,
                            num_channels=1,
                        ) as f:
                            f.write(augmented_audio)
                    mp3_path = aa_mp3_path
                if self.use_demucs:
                    # XXX: ultimately not used! Since not robust to attacks.
                    if not mp3_path:
                        raise ValueError(f"{d['id']} has no mp3 paths! {d}")

                    if "fake" in d["raw_label"]:
                        demucs_dir = "demucs/fake"
                        song_id = mp3_path.split("/")[-1].split(".")[0]
                    elif "real" in d["raw_label"]:
                        demucs_dir = "demucs/real"
                        song_id = d["id"]
                    else:
                        raise ValueError(f"Unknown label: {d['raw_label']}, {d}")

                    if attack_to_apply:
                        demucs_dir = os.path.join(demucs_dir, attack_to_apply)

                    try:
                        mp3_path = separate_audio(
                            mp3_path, song_id, demucs_dir, "cuda:0"
                        )
                    except Exception:
                        raise ValueError(f"Could not separate audio for {d['id']}, {d}")

                if self.use_vad:
                    # XXX: ultimately not used! Since not robust to attacks.
                    if not mp3_path:
                        raise ValueError(f"{d['id']} has no mp3 paths! {d}")
                    if "fake" in d["raw_label"]:
                        vad_dir = "vad_output/fake"
                    elif "real" in d["raw_label"]:
                        vad_dir = "vad_output/real"
                    else:
                        raise ValueError(f"Unknown label: {d['raw_label']}, {d}")
                    if attack_to_apply:
                        vad_dir = os.path.join(vad_dir, attack_to_apply)
                    # XXX: if using demucs, this is a WAV file but so it will be loaded as _vocal.wav
                    # and this will do nothing (but not an issue)
                    new_mp3_path = mp3_path.replace(".mp3", "_speech.mp3").split("/")[
                        -1
                    ]
                    vad_out_path = os.path.join(vad_dir, new_mp3_path)
                    if not os.path.exists(vad_dir):
                        os.makedirs(vad_dir, exist_ok=True)
                    if os.path.isfile(vad_out_path):
                        # print(f"VAD output already exists for {d['id']}")
                        # check if file size is greater than 10000 bytes; a few early runs failed
                        if os.path.getsize(vad_out_path) < 10000:
                            print(f"VAD output too small for {d['id']}, re-running")
                        else:
                            mp3_path = vad_out_path
                    else:
                        wav = read_audio(mp3_path)
                        speech_timestamps = get_speech_timestamps(
                            wav,
                            vad,
                            return_seconds=True,  # in seconds
                        )
                        try:
                            # to milliseconds
                            clip_times = [
                                {
                                    "start": int(s["start"] * 1000),
                                    "end": int(s["end"] * 1000),
                                }
                                for s in speech_timestamps
                            ]

                            # save to mp3
                            initial_audio_segment = AudioSegment.from_file(
                                mp3_path
                            ).set_frame_rate(16000)

                            if len(clip_times) == 0:
                                # no speech detected, keep full audio
                                clip_times = [
                                    {"start": 0, "end": len(initial_audio_segment)}
                                ]
                            new_audio = AudioSegment.empty()
                            for clip_time in clip_times:
                                new_audio += initial_audio_segment[
                                    clip_time["start"] : clip_time["end"]
                                ]
                            new_audio.export(vad_out_path, format="mp3", bitrate="128k")
                            # print(f"Saved VAD output to {vad_out_path}")
                            mp3_path = vad_out_path
                        except Exception:
                            print(clip_times, mp3_path)
                            raise ValueError(
                                f"Could not get speech timestamps for {d['id']}: {d}"
                            )

                # LOAD LOGPROBS, before computing embeddings
                if self.logprob_dir:
                    song_logprobs = logprobs[idx]
                    assert logprobs[idx]["id"] == d["id"], (
                        f"ID mismatch between logprobs and dataset for {subset}",
                        logprobs["id"],
                        d["id"],
                    )
                else:
                    song_logprobs = None

                # EMBED!
                song_vector = self.compute_embeddings(
                    model, lyrics, song_logprobs, mp3_path
                )

                embs[subset].append(song_vector)

                # Store with full metadata
                all_vectors[vectors_subset][vector_key] = {
                    "vectors": song_vector,
                    "raw_label": d["raw_label"],
                    "id": d["id"],
                    "model_name": d["model_name"],
                }
                updated_elements_cpt += 1

            if updated_elements_cpt > 0:
                dataset[subset] = dataset[subset].map(
                    lambda example, idx: {"embeddings": embs[subset][idx]},
                    with_indices=True,
                )

        # only save again if new vectors were computed and not loaded
        if updated_elements_cpt > 0:
            if not os.path.exists(os.path.dirname(path_vectors)):
                os.makedirs(os.path.dirname(path_vectors))
            with open(path_vectors, "w") as f_vectors:
                json.dump(all_vectors, f_vectors, indent=4)
            print(f"Saved vectors to {path_vectors}")

            if save_full_hf_dataset:
                dataset_save_path = f"{self.base_dir}/dataset/{model_type_str}/{self.save_embeddings_dir}_{self.model_variant_short}"

                dataset.save_to_disk(dataset_save_path)
                print(f"Saved full dataset to {dataset_save_path}")

        if do_save_and_classify:
            model_name, dataset_name, config_name = self.save_dir.split("/")

            results_save_dir = os.path.join(
                self.base_dir,
                self.classifier,
                self.protocol,
                dataset_name,
                self.model_type,
                model_name,
                config_name,
            )
            self.classify_and_save(
                results_save_dir,
                embs,
                labels,
                modelnames,
                artists,
                langs,
                genres,
                lyrics_id,
            )
            return results_save_dir
        else:
            return embs, labels, modelnames, artists, langs, genres, lyrics_id

    def classify_and_save(
        self,
        model_save_dir,
        _embs,
        _labels,
        _modelnames,
        _artists,
        _langs,
        _genres,
        _lyrics_id,
    ):
        if not os.path.exists(os.path.dirname(model_save_dir)):
            os.makedirs(os.path.dirname(model_save_dir))
        if not self.predict_only:
            if self.classifier == "knn":
                neigh = KNeighborsClassifier(n_neighbors=self.n_neighbors)
                # save model

            elif self.classifier == "mlp":
                # TODO: train on regular, eval on attacked audio
                neigh = MLPClassifier(
                    hidden_layer_sizes=[1028, 512],
                    activation="relu",
                    solver="adam",
                    random_state=42,
                    verbose=True,
                    max_iter=1000,
                    # batch_size=32,
                    learning_rate="adaptive",
                )
                # standard scaler
                scaler = StandardScaler()
                train_data = scaler.fit_transform(_embs[self.CURRENT_TRAIN_SUBSET])
                test_data = scaler.transform(_embs[self.CURRENT_SUBSET])
            elif self.classifier == "lit_mlp":
                if self.model_type == "ensemble":
                    embedding_shapes = [
                        emb.shape[1] for emb in _embs[self.CURRENT_TRAIN_SUBSET]
                    ]

                    train_data = np.concatenate(
                        _embs[self.CURRENT_TRAIN_SUBSET], axis=1
                    )
                    test_data = np.concatenate(_embs[self.CURRENT_SUBSET], axis=1)
                else:
                    # Scale the data
                    scaler = StandardScaler()
                    self.scalers = scaler
                    train_data = scaler.fit_transform(_embs[self.CURRENT_TRAIN_SUBSET])
                    test_data = scaler.transform(_embs[self.CURRENT_SUBSET])
                    # train_data = _embs[self.CURRENT_TRAIN_SUBSET]
                    # test_data = _embs[self.CURRENT_SUBSET]

                # Convert to tensors
                X_train = torch.FloatTensor(train_data)
                y_train = torch.LongTensor(_labels[self.CURRENT_TRAIN_SUBSET])

                # Create data loaders
                train_dataset = TensorDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

                # Initialize and train model
                if self.model_type == "ensemble":
                    model = TwoStreamLitMLP(embedding_sizes=embedding_shapes)
                else:
                    model = LitMLP(input_size=X_train.shape[1])

                trainer = pl.Trainer(max_epochs=100, enable_progress_bar=True)
                trainer.fit(model, train_loader)
                # save model
                trainer.save_checkpoint(f"{model_save_dir}.ckpt")
            else:
                raise ValueError(
                    f"The classifier type {self.classifier} is not supported!"
                )
            if isinstance(self.scalers, list):
                for i, scaler in enumerate(self.scalers):
                    with open(f"{model_save_dir}_scaler_{i}.pkl", "wb") as f:
                        joblib.dump(scaler, f)
                        print(
                            f"Saved sklearn scaler to {model_save_dir}_scaler_{i}.pkl"
                        )
            else:
                with open(f"{model_save_dir}_scaler.pkl", "wb") as f:
                    joblib.dump(self.scalers, f)
                    print(f"Saved sklearn scaler to {model_save_dir}_scaler.pkl")
        else:
            # XXX: this only works with 1 attack at a time!
            attack_to_apply = None
            for effect in AUDIO_ATTACKS:
                if effect in self.save_embeddings_dir:
                    attack_to_apply = effect
            if attack_to_apply:
                model_load_dir = model_save_dir.replace(attack_to_apply + "_", "")
            else:
                model_load_dir = model_save_dir
            model_load_dir = (
                model_load_dir.replace("UDIO-", "")
                .replace("_DEFAULT", "")
                .replace("-aa", "")
            )
            if "_D" in model_load_dir and "DEMUCS" not in model_load_dir:
                model_load_dir = model_load_dir.replace("_D", "")

            if self.classifier != "lit_mlp":
                with open(f"{model_load_dir}.pkl", "rb") as f:
                    neigh = joblib.load(f)
                    print(f"Loaded sklearn model from {model_load_dir}.pkl")
            elif self.model_type == "ensemble":
                model = TwoStreamLitMLP.load_from_checkpoint(f"{model_load_dir}.ckpt")
                scalers = []
                while os.path.isfile(f"{model_load_dir}_scaler_{len(scalers)}.pkl"):
                    with open(f"{model_load_dir}_scaler_{len(scalers)}.pkl", "rb") as f:
                        scaler = joblib.load(f)
                        scalers.append(scaler)
                self.scalers = scalers
            else:
                model = LitMLP.load_from_checkpoint(f"{model_load_dir}.ckpt")
                with open(f"{model_load_dir}_scaler.pkl", "rb") as f:
                    scaler = joblib.load(f)
                    self.scalers = scaler

        if self.classifier != "lit_mlp":
            if self.model_type == "ensemble":
                test_data = np.concatenate(_embs[self.CURRENT_SUBSET], axis=1)
                if not self.predict_only:
                    train_data = np.concatenate(
                        _embs[self.CURRENT_TRAIN_SUBSET], axis=1
                    )
            else:
                test_data = _embs[self.CURRENT_SUBSET]
                if not self.predict_only:
                    train_data = _embs[self.CURRENT_TRAIN_SUBSET]
            if not self.predict_only:
                neigh.fit(
                    train_data,
                    _labels[self.CURRENT_TRAIN_SUBSET],
                )

            all_preds_method = []
            all_probas_method = []

            for d in tqdm(test_data):
                probas = neigh.predict_proba([d])[0].tolist()
                all_probas_method.append(probas)
                max_value = max(probas)
                idx_max = probas.index(max_value)
                all_preds_method.append(idx_max)
            with open(f"{model_save_dir}.pkl", "wb") as f:
                joblib.dump(neigh, f)
                print(f"Saved sklearn model to {model_save_dir}.pkl")
        else:
            if self.predict_only:
                if self.model_type == "ensemble":
                    test_data = []
                    for test_embedding, scaler in zip(
                        _embs[self.CURRENT_SUBSET],
                        self.scalers,
                    ):
                        test_data.append(scaler.transform(test_embedding))
                    test_data = np.concatenate(test_data, axis=1)
                else:
                    test_data = self.scalers.transform(_embs[self.CURRENT_SUBSET])
            else:
                if self.model_type == "ensemble":
                    test_data = np.concatenate(_embs[self.CURRENT_SUBSET], axis=1)
                else:
                    test_data = self.scalers.transform(_embs[self.CURRENT_SUBSET])
            X_test = torch.FloatTensor(test_data)
            y_test = torch.LongTensor(_labels[self.CURRENT_SUBSET])
            test_dataset = TensorDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

            all_preds_method = []
            all_probas_method = []

            # batched predictions
            for batch in test_loader:
                X_batch, _ = batch
                logits = model(X_batch)
                probas = model.softmax(logits).tolist()
                all_probas_method.extend(probas)
                preds = torch.argmax(logits, dim=1).tolist()
                all_preds_method.extend(preds)

        with open(f"{model_save_dir}.json", "w") as f:
            json.dump(
                {
                    "predictions": all_preds_method,
                    "probabilities": all_probas_method,
                    "references": _labels[self.CURRENT_SUBSET],
                    "model_names": _modelnames[self.CURRENT_SUBSET],
                    "artists": _artists[self.CURRENT_SUBSET],
                    "langs": _langs[self.CURRENT_SUBSET],
                    "genres": _genres[self.CURRENT_SUBSET],
                    "lyrics_ids": _lyrics_id[self.CURRENT_SUBSET],
                },
                f,
                indent=4,
            )
            print(f"Saved results to {model_save_dir}.json")

        CONFIG = {
            "hyps": all_preds_method,
            "refs": _labels[self.CURRENT_SUBSET],
            "probs": all_probas_method,
            "meta": [
                (art, lang, gen, mname)
                for art, lang, gen, mname in zip(
                    _artists[self.CURRENT_SUBSET],
                    _langs[self.CURRENT_SUBSET],
                    _genres[self.CURRENT_SUBSET],
                    _modelnames[self.CURRENT_SUBSET],
                )
            ],
        }

        self.get_metrics(
            CONFIG["hyps"], CONFIG["refs"], CONFIG["meta"], model_path=model_save_dir
        )

    def get_metrics(
        self, _combined_classes, _combined_refs, _combined_meta, model_path
    ):
        logs_model_path = model_path.replace(".json", "").replace("knn", "knn_logs")
        print(logs_model_path)
        os.makedirs(logs_model_path.rsplit("/", 1)[0], exist_ok=True)
        # global f1
        cr = classification_report(
            _combined_refs,
            _combined_classes,
            digits=3,
            zero_division=0,
            output_dict=False,
        )
        print(cr)

        with open(logs_model_path + "_summary.txt", "w") as f:
            f.write(cr)
        print(f"Saved logs to {model_path + '_summary.txt'}")

        # granular f1
        models_names = list(set([m[3] for m in _combined_meta]))
        if len(models_names) == 4:
            models_names = ["mistral", "tinyllama", "wizardlm2", "original"]
        elif len(models_names) == 3:
            models_names = ["tinyllama", "wizardlm2", "original"]

        if self.protocol == "nlp4musa":
            results = {}

            for hyp, ref, meta in zip(
                _combined_classes, _combined_refs, _combined_meta
            ):
                artist, lang, genre, modelname = meta

                if modelname not in results:
                    results[modelname] = {}

                if lang not in results[modelname]:
                    results[modelname][lang] = {"hyps": [], "refs": []}

                results[modelname][lang]["hyps"].append(hyp)
                results[modelname][lang]["refs"].append(ref)

            results_f1 = {}

            langs = []

            for modelname in results:
                print(
                    modelname.replace("generations_generations_", "").replace(
                        "_normalized.json", ""
                    )
                )

                results_f1[modelname] = {}

                for lang in sorted(list(results[modelname])):
                    cr = classification_report(
                        results[modelname][lang]["refs"],
                        results[modelname][lang]["hyps"],
                        digits=4,
                        output_dict=True,
                        zero_division=0,
                    )

                    if len(models_names) == 1:
                        lang_score_f1 = f"{cr['0']['recall'] * 100:.2f},{cr['1']['recall'] * 100:.2f}"
                    elif "original" in modelname:
                        lang_score_f1 = cr["0"]["recall"] * 100
                    else:
                        lang_score_f1 = cr["1"]["recall"] * 100
                    if isinstance(lang_score_f1, float):
                        print(lang, " - ", "{:.2f}".format(lang_score_f1))
                        results_f1[modelname][lang] = "{:.2f}".format(lang_score_f1)
                    else:
                        print(lang, " - ", lang_score_f1)
                        results_f1[modelname][lang] = lang_score_f1
                    langs.append(lang)

                print()

            langs = sorted(list(set(langs)))

            output_table = []

            output_table.append(
                " & ".join(
                    [""]
                    + [
                        m.replace("generations_generations_", "").replace(
                            "_normalized.json", ""
                        )
                        for m in models_names
                    ]
                )
                + " \\\\"
            )

            for lang in langs:
                output_table.append(
                    lang
                    + " & "
                    + " & ".join(
                        [
                            (
                                results_f1[modelname][lang]
                                if lang in results_f1[modelname]
                                else "-1"
                            )
                            for modelname in models_names
                        ]
                    )
                    + " \\\\"
                )

            all_avg_models = []

            if len(models_names) > 1:
                for modelname in models_names:
                    values = [
                        float(results_f1[modelname][v]) for v in results_f1[modelname]
                    ]
                    avg = sum(values) / len(values)
                    all_avg_models.append(avg)

                output_table.append(
                    "AVG & "
                    + " & ".join(["{:.2f}".format(avm) for avm in all_avg_models])
                    + " \\\\"
                )

                # old average: simple average of all models
                overall_old_average = sum(all_avg_models) / len(all_avg_models)

                # new average: average of all models except the last one, but
                # but adds back last model, so gets double-weighted vs. other models.
                overall_new_average = (
                    (sum(all_avg_models[:-1]) / len(all_avg_models[:-1]))
                    + all_avg_models[-1]
                ) / 2

                output_table.append(
                    "Old AVG & "
                    + " & ".join([" " for _ in range(len(all_avg_models) - 1)])
                    + " & "
                    + "{:.2f}".format(overall_old_average)
                    + " \\\\"
                )
                output_table.append(
                    "New AVG & "
                    + " & ".join([" " for _ in range(len(all_avg_models) - 1)])
                    + " & "
                    + "{:.2f}".format(overall_new_average)
                    + " \\\\"
                )

            output_table = "\n".join(output_table)
            print(output_table)

            f_out_table = open(logs_model_path + "_full.txt", "w")
            f_out_table.write(output_table)
            f_out_table.close()
            print(f"Saved logs to {logs_model_path + '_full.txt'}")

        elif self.protocol == "emnlp":
            raise ValueError("The protocol EMNLP is not supported for now!.")
