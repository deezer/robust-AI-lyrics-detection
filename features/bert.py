from typing import List, Optional

from feature_extractor import FeatureExtractor
from sentence_transformers import SentenceTransformer


class Bert(FeatureExtractor):
    def __init__(
        self,
        dataset,
        save_dir: Optional[str] = None,
        save_embeddings_dir: Optional[str] = None,
        max_tokens: Optional[int] = 512,
        hf_token: Optional[str] = None,
        classifier: Optional[str] = None,
        protocol: Optional[str] = None,
        current_train_subset: Optional[str] = None,
        current_subset: str = "test",
        n_neighbors: int = 3,
        bert_variant: str = "all-mpnet-base-v2",
        **kwargs
    ):
        self.bert_variant = bert_variant
        self.bert_variant_short = bert_variant.split("/")[-1]
        super().__init__(
            dataset,
            save_dir=save_dir,
            save_embeddings_dir=save_embeddings_dir,
            max_tokens=max_tokens,
            hf_token=hf_token,
            classifier=classifier,
            protocol=protocol,
            current_train_subset=current_train_subset,
            current_subset=current_subset,
            n_neighbors=n_neighbors,
            **kwargs
        )
        self.model_type = "bert"
        self.model_variant = bert_variant
        self.model_variant_short = self.bert_variant_short

    def load_model(self):
        return SentenceTransformer(self.bert_variant, device="cuda:0")

    def compute_embeddings(
        self, model, lyrics: str, logprobs=None, mp3_path=None, **kwargs
    ) -> List[float]:
        lyric_text = (
            lyrics.lower()
            .replace("(", " ")
            .replace(")", " ")
            .replace(",", " ")
            .replace("'", " ")
            .replace("-", " ")
            .replace("!", " ")
            .replace("?", " ")
        )
        lyrics_vector = model.encode(lyric_text, precision="float32")
        return lyrics_vector.tolist()
