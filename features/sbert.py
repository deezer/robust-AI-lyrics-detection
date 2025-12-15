from typing import List, Optional

from feature_extractor import FeatureExtractor
from sentence_transformers import SentenceTransformer


class SentenceBert(FeatureExtractor):
    def __init__(
        self,
        dataset,
        save_dir,
        save_embeddings_dir,
        model_variant: str = "paraphrase-MiniLM-L6-v2",
        max_tokens: Optional[int] = 512,
        hf_token: Optional[str] = None,
        classifier: Optional[str] = None,
        protocol: Optional[str] = None,
        current_train_subset: str = "train",
        current_subset: str = "test",
        n_neighbors: int = 3,
        predict_only: bool = False,
    ):
        self.model_variant = model_variant
        self.model_variant_short = self._get_model_variant_short()
        self.model_type = "sbert"

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
        model = SentenceTransformer(
            self.model_variant,
            device="cuda:0",
            trust_remote_code=True,
            model_kwargs={"torch_dtype": "float16"},
        )
        return model

    def compute_embeddings(
        self, model, lyrics: str, logprobs=None, mp3_path=None, **kwargs
    ) -> List[float]:
        lyrics_vector = model.encode(
            lyrics, normalize_embeddings=True, precision="float32"
        )
        return lyrics_vector
