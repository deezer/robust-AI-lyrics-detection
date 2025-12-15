import os
from pprint import pprint

from predict import NLL_MODELS, Args, predict

BASE_DIR = os.environ.get("PROJECT_BASE_DIR", "")

DATA_FILENAMES = {
    "real": "real_songs.json",
    "fake": "fake_songs.json",
    "real_fake": "halffake_songs.json",
}

# use output filenames from transcribers, but without _N{N} suffix
# needs to be available for both classes!
TRANSCRIPTION_MODELS = [
    "fwhisper-large-v2_DEFAULT",
    # "fwhisper-large-v3_DEFAULT",
    # "fwhisper-large-v3-turbo_DEFAULT",
    # "sM4T-L_Wv3-LANG",
    # "mms_Wv3-LANG",
]

STRATIFY_MODELS = False
# To train on lyrics from specific LLMs and eval on another, set e.g.:
# STRATIFY_MODELS = ["wizardlm2", "mistral"]


DATA_COMBOS = []

# TODO: add back real_only lyrics case
for transcription_model in TRANSCRIPTION_MODELS:
    # idx 0,1 since both Suno and Udio always generate 2 songs
    for generation_idx in range(0, 1):
        # main scenario: real vs. fake, using transcripts (T... transcript)
        DATA_COMBOS.append(
            {
                "name": f"realT_fakeT_{transcription_model}_{generation_idx}",
                "real_transcribed": f"output/{transcription_model}",
                "fake_transcribed": f"output_fake/{transcription_model}",
                "generation_idx": generation_idx,
            }
        )
        # the following are only used for a subset of models:
        # half-fake vs. fake
        # ACL Table 3
        # DATA_COMBOS.append(
        #     {
        #         "name": f"realfakeT_fakeT_{transcription_model}_{generation_idx}",
        #         "real_fake_transcribed": f"output_real_fake/{transcription_model}",
        #         "fake_transcribed": f"output_fake/{transcription_model}",
        #         "generation_idx": generation_idx,
        #     },
        # )
        # half-fake vs. real
        # # DATA_COMBOS.append(
        #     {
        #         "name": f"realT_realfakeT_{transcription_model}_{generation_idx}",
        #         "real_transcribed": f"output/{transcription_model}",
        #         "real_fake_transcribed": f"output_real_fake/{transcription_model}",
        #         "generation_idx": generation_idx,
        #     },
        # )


# for ISMIR Table 3
if STRATIFY_MODELS:
    DATA_COMBOS = [
        {
            "name": f"realT_fakeT-{'-'.join(STRATIFY_MODELS)}-train125_{transcription_model}_{generation_idx}",
            "real_transcribed": f"output/{transcription_model}",
            "fake_transcribed": f"output_fake/{transcription_model}",
            "generation_idx": generation_idx,
        }
    ]
pprint(DATA_COMBOS)
print("\n\n\n")

# structure: {model_type_key: [model1, model2, ...]}
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
    # ratio 0.5:0.5 only matters for kNN by scaling embeddings. Not relevant for MLP.
    # format is [model_type_key: model1, model_type_key: model2]
    "ensemble_0.5:0.5": [
        [
            "llm2vec:McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "xeus:128",
        ],
        [
            "llm2vec:McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "sbert:BAAI/bge-multilingual-gemma2",
        ],
        [
            "llm2vec:McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "luar:LUAR-MUD",
        ],
        [
            "sbert:BAAI/bge-multilingual-gemma2",
            "luar:LUAR-MUD",
        ],
        [
            "xeus:128",
            "w2v2:128",
        ],
        [
            "xeus:128",
            "mms:128",
        ],
        [
            "w2v2:128",
            "mms:128",
        ],
        [
            "llm2vec:McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "w2v2:128",
        ],
        [
            "llm2vec:McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "mms:128",
        ],
        [
            "sbert:BAAI/bge-multilingual-gemma2",
            "xeus:128",
        ],
        [
            "sbert:BAAI/bge-multilingual-gemma2",
            "w2v2:128",
        ],
        [
            "sbert:BAAI/bge-multilingual-gemma2",
            "mms:128",
        ],
        [
            "luar:LUAR-MUD",
            "xeus:128",
        ],
        [
            "luar:LUAR-MUD",
            "w2v2:128",
        ],
        [
            "luar:LUAR-MUD",
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
                (real_dataset_name == "fake" or fake_dataset_name == "fake")
                and ("model_key" == "xeus" or "ensemble" in model_key)
                or (
                    data_combo["name"] == "real_only"
                    and (model_key == "xeus" or "ensemble" in model_key)
                )
            ):
                print("Skipping fake dataset for XEUS")
                continue
            if STRATIFY_MODELS and "fakeT" not in data_combo["name"]:
                raise ValueError("Model stratification only done for fakeT")

            # XXX: in some cases, some transcripts were run with 5, some with 1 run. Here, we take whichever is available.
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

            if data_combo[fake_dataset_name] == "":
                real_dataset_path = data_combo[real_dataset_name]
                fake_dataset_path = ""
            else:
                # Get appropriate filenames for real and fake datasets
                real_dataset_type = determine_dataset_type(
                    data_combo[real_dataset_name]
                )
                fake_dataset_type = determine_dataset_type(
                    data_combo[fake_dataset_name]
                )

                real_dataset_filename = DATA_FILENAMES[real_dataset_type]
                fake_dataset_filename = DATA_FILENAMES[fake_dataset_type]

                # Find valid paths for both datasets
                real_dataset_path = find_existing_dataset_path(
                    BASE_DIR, data_combo[real_dataset_name], real_dataset_filename
                )
                fake_dataset_path = find_existing_dataset_path(
                    BASE_DIR, data_combo[fake_dataset_name], fake_dataset_filename
                )
                if STRATIFY_MODELS:
                    fake_dataset_path = fake_dataset_path.replace(
                        ".json", f"_{'-'.join(STRATIFY_MODELS)}-train125.json"
                    )
            # replace "DEFAULT" name with "D"
            if "DEFAULT" in data_combo["name"]:
                data_combo["name"] = (
                    data_combo["name"].replace("DEFAULT", "").replace("__", "_")
                )

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
            _, all_data = predict(current_args, all_data)
            old_data_combo = data_combo
