import os
from pprint import pprint

from predict import NLL_MODELS, Args, predict

BASE_DIR = os.environ.get("PROJECT_BASE_DIR", "")
DATA_FILENAMES = {
    "real": "real_songs.json",
    "fake": "fake_songs.json",
    "real_fake": "halffake_songs.json",
}

DATA_FILENAMES_UDIO = {
    "real": "real_songs.json",
    "fake": "udio_songs.json",
}

# Table 4; we did this with whisper-large-v2
TRANSCRIPTION_MODELS = [
    # audio attacks/perturbations
    "fwhisper-large-v2_pitch",
    "fwhisper-large-v2_reverb",
    "fwhisper-large-v2_eq",
    "fwhisper-large-v2_noise",
    "fwhisper-large-v2_stretch",
    # UDIO
    "fwhisper-large-v2_DEFAULT",
]


# always set to true for final experiments (leaver real songs intact, i.e., non-perturbed)
ATTACK_ONLY_FAKE = True


if ATTACK_ONLY_FAKE:
    SUFFIX = "-aa"
else:
    SUFFIX = ""

DATA_COMBOS = []

for transcription_model in TRANSCRIPTION_MODELS:
    for generation_idx in range(0, 2):
        # XXX: assumes DEFAULT is in the name of transcriber, which is not necessarily the case, e.g., for demucs (handled below)
        if "DEFAULT" not in transcription_model:
            DATA_COMBOS.append(
                {
                    "name": f"realT_fakeT{SUFFIX}_{transcription_model}_{generation_idx}",
                    "real_transcribed": f"output/{transcription_model}",
                    "fake_transcribed": f"output_fake/{transcription_model}",
                    "generation_idx": generation_idx,
                }
            )
        else:
            # UDIO
            DATA_COMBOS.append(
                {
                    "name": f"UDIO-realT_fakeT_{transcription_model}_{generation_idx}",
                    "real_transcribed": f"output_udio/{transcription_model}",
                    "fake_transcribed": f"output_fake_udio/{transcription_model}",
                    "generation_idx": generation_idx,
                }
            )
pprint(DATA_COMBOS)
print("\n\n\n")

MODELS = {
    "sbert": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-m3",
        "BAAI/bge-multilingual-gemma2",
    ],
    "luar": ["LUAR-MUD", "LUAR-CRUD"],
    "llm2vec": [
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    ],
    "entropy": [
        "meta-llama/Meta-Llama-3-8B",
    ],
    "max_nll": [
        "meta-llama/Meta-Llama-3-8B",
    ],
    "perplexity": [
        "meta-llama/Meta-Llama-3-8B",
    ],
    "mink_10": [
        "meta-llama/Meta-Llama-3-8B",
    ],
    "xeus": [
        "128",
    ],
    "w2v2": [
        "128",
    ],
    "mms": [
        "128",
    ],
    "ensemble_0.5:0.5": [
        [
            "llm2vec:McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "xeus:128",
        ],
        [
            "llm2vec:McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "w2v2:128",
        ],
        [
            "llm2vec:McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "mms:128",
        ],
    ],
}

pprint(MODELS)
print("\n\n\n")

old_data_combo = None
for data_combo in DATA_COMBOS:
    for model_key in MODELS:
        for model in MODELS[model_key]:
            print(
                "RUNNING: ",
                data_combo,
                model,
                model_key,
                "\n",
                "-" * 50,
                "\n",
                data_combo["name"],
                "\n",
                "-" * 50,
                "\n",
            )

            real_dataset_name, fake_dataset_name = (
                list(data_combo.keys())[1],
                list(data_combo.keys())[2],
            )
            if (
                real_dataset_name == "fake"
                or fake_dataset_name == "fake"
                and "model_key" == "xeus"
            ):
                print("Skipping fake dataset for XEUS")
                continue

            # XXX: some transcripts were run with 5, some with 1 run. Here, we take whichever is available.
            def determine_dataset_type(directory_path: str) -> str:
                """
                Determine which dataset type (real, fake, or real_fake) based on directory path.
                """
                if "real_fake" in directory_path:
                    return "real_fake"
                elif "fake" in directory_path or "generated" in directory_path:
                    return "fake"
                return "real"

            def find_existing_dataset_path(
                base_dir: str, dataset_dir: str, filename: str
            ) -> str:
                """
                Find the first existing dataset path by trying different N values.
                Returns the first valid path or raises ValueError if none exist.
                """
                # None in case of non-transcribed
                possible_n_values = [5, 1, None]
                for n in possible_n_values:
                    suffix = f"_N{n}" if n else ""
                    path = f"{base_dir}{dataset_dir}{suffix}/{filename}"
                    if os.path.exists(path):
                        return path

                raise ValueError(f"No valid dataset path found for {dataset_dir}")

            # Get appropriate filenames for real and fake datasets
            if "UDIO" in data_combo["name"]:
                real_dataset_filename = DATA_FILENAMES_UDIO["real"]
                fake_dataset_filename = DATA_FILENAMES_UDIO["fake"]
                # udio has no real_fake anyways
            else:
                real_dataset_type = determine_dataset_type(
                    data_combo[real_dataset_name]
                )
                fake_dataset_type = determine_dataset_type(
                    data_combo[fake_dataset_name]
                )

                real_dataset_filename = DATA_FILENAMES[real_dataset_type]
                fake_dataset_filename = DATA_FILENAMES[fake_dataset_type]

            # Find valid paths for both datasets
            if ATTACK_ONLY_FAKE:
                data_combo[real_dataset_name] = (
                    data_combo[real_dataset_name]
                    .replace("pitch", "DEFAULT")
                    .replace("reverb", "DEFAULT")
                    .replace("eq", "DEFAULT")
                    .replace("noise", "DEFAULT")
                    .replace("stretch", "DEFAULT")
                )
                if "UDIO" in data_combo["name"]:
                    raise ValueError("UDIO not supported for ATTACK_ONLY_FAKE")
            real_dataset_path = find_existing_dataset_path(
                BASE_DIR,
                data_combo[real_dataset_name],
                real_dataset_filename,
            )
            fake_dataset_path = find_existing_dataset_path(
                BASE_DIR,
                data_combo[fake_dataset_name],
                fake_dataset_filename,
            )
            if "DEFAULT" in data_combo["name"]:
                data_combo["name"] = data_combo["name"].replace("DEFAULT", "D")

            # re-use already loaded data if same
            if old_data_combo != data_combo:
                all_data = None
            current_args = Args(
                protocol="nlp4musa",
                dataset_combo_name=data_combo["name"],
                real_dataset_name=list(data_combo.keys())[1],
                fake_dataset_name=list(data_combo.keys())[2],
                only_real=True if list(data_combo.values())[2] == "" else False,
                real_dataset=real_dataset_path,
                fake_dataset=fake_dataset_path,
                detector=model_key if model_key not in NLL_MODELS else "loglikelihood",
                detector_model=model,
                strategy=None if model_key not in NLL_MODELS else model_key,
                classifier="lit_mlp",
                n_neighbors=3,
                max_tokens=512,
                postprocess=False,
                generation_idx=data_combo["generation_idx"],
            )
            _, all_data = predict(current_args, all_data, predict_only=True)
            old_data_combo = data_combo
