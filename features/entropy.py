from typing import List, Optional

from feature_extractor import FeatureExtractor
from scipy.stats import entropy


class Entropy(FeatureExtractor):
    def __init__(
        self,
        dataset,
        save_dir,
        save_embeddings_dir,
        hf_token: Optional[str] = None,
        classifier: Optional[str] = None,
        protocol: Optional[str] = None,
        current_train_subset: str = "train",
        current_subset: str = "test",
        n_neighbors: int = 3,
        logprob_dir: Optional[str] = None,
        logprob_model: Optional[str] = "meta-llama/Meta-Llama-3.1-8B",
    ):
        # for consistency, we use the logprob_dir as the save_dir
        self.logprob_dir = logprob_dir
        self.model_variant = logprob_model
        self.model_variant_short = self._get_model_variant_short()
        self.model_type = "entropy"

        super().__init__(
            dataset,
            save_dir=self.model_variant_short + "/" + save_dir,
            save_embeddings_dir=save_embeddings_dir,
            hf_token=hf_token,
            classifier=classifier,
            protocol=protocol,
            current_train_subset=current_train_subset,
            current_subset=current_subset,
            n_neighbors=n_neighbors,
            logprob_dir=logprob_dir,
            logprob_model=logprob_model,
        )

    def load_model(self):
        return

    def compute_embeddings(
        self, model, lyrics: str, logprobs=None, mp3_path=None, **kwargs
    ) -> List[float]:
        if logprobs is None:
            raise ValueError(
                "logprobs must be provided for entropy feature extraction."
            )
        all_ppl_tokens = logprobs["tokens_log_probs"]
        ppl_entropy = [entropy(all_ppl_tokens, base=2)]
        return ppl_entropy
