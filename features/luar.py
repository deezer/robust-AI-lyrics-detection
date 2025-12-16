from typing import List, Optional

from feature_extractor import FeatureExtractor
from transformers import AutoModel, AutoTokenizer


class Luar(FeatureExtractor):
    def __init__(
        self,
        dataset,
        save_dir,
        save_embeddings_dir,
        model_variant: str = "LUAR-CRUD",
        max_tokens: Optional[int] = 512,
        hf_token: Optional[str] = None,
        classifier: Optional[str] = None,
        protocol: Optional[str] = None,
        current_train_subset: str = "train",
        current_subset: str = "test",
        n_neighbors: int = 3,
        predict_only: bool = False,
    ):
        self.model_variant = model_variant.upper()
        self.model_variant_short = self.model_variant.split("-")[-1]
        self.model_type = "luar"
        if self.model_variant_short not in ["MUD", "CRUD"]:
            raise ValueError(
                f"Error, the LUAR variant doesn't exist or not allowed! ({self.model_variant_short})"
            )

        super().__init__(
            dataset,
            save_dir=self.model_variant_short + "/" + save_dir,
            save_embeddings_dir=save_embeddings_dir,
            max_tokens=max_tokens,
            hf_token=hf_token,
            classifier=classifier,
            protocol=protocol,
            current_train_subset=current_train_subset,
            current_subset=current_subset,
            n_neighbors=n_neighbors,
            predict_only=predict_only,
        )

    def load_model(self):
        model_name = f"rrivera1849/{self.model_variant}"
        print(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.to("cuda")
        return model, tokenizer

    def compute_embeddings(
        self, model, lyrics: str, logprobs=None, mp3_path=None, **kwargs
    ) -> List[float]:
        model, tokenizer = model

        tokenized_text = tokenizer(
            lyrics,
            max_length=int(self.max_tokens),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        tokenized_text["input_ids"] = tokenized_text["input_ids"].reshape(
            kwargs.get("batch_size", 1), kwargs.get("episode_length", 1), -1
        )
        tokenized_text["attention_mask"] = tokenized_text["attention_mask"].reshape(
            kwargs.get("batch_size", 1), kwargs.get("episode_length", 1), -1
        )
        tokenized_text.to("cuda")

        out = model(**tokenized_text)
        lyrics_vector = out[0].tolist()
        return lyrics_vector
