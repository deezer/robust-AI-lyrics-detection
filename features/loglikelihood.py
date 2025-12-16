import math
from typing import List, Optional

from feature_extractor import FeatureExtractor
from scipy.stats import entropy


class LogLikelihood(FeatureExtractor):
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
        strategy: Optional[str] = "mink",
        predict_only: bool = False,
    ):
        # for consistency, we use the logprob_dir as the save_dir
        self.logprob_dir = logprob_dir
        self.model_variant = logprob_model
        self.model_variant_short = self._get_model_variant_short()
        self.model_type = "loglikelihood"

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
            strategy=strategy,
            predict_only=predict_only,
        )

    def load_model(self):
        return

    def compute_embeddings(
        self, model, lyrics: str, logprobs=None, mp3_path=None, **kwargs
    ) -> List[float]:
        if logprobs is None:
            raise ValueError(
                "logprobs must be provided for loglikelihood feature extraction."
            )
        all_ppl_tokens = logprobs["tokens_log_probs"]
        if self.strategy == "perplexity":
            features = [math.exp(sum(all_ppl_tokens) / len(all_ppl_tokens))]
        elif "mink" in self.strategy:
            top_mink_size = int(self.strategy.split("_")[1])
            features = sorted(all_ppl_tokens, reverse=True)
            top_mink_size = max(1, int(len(features) * top_mink_size / 100))
            features = features[0:top_mink_size]
            features = [sum(features) / len(features)]
        elif self.strategy == "max_nll":
            features = [max(all_ppl_tokens)]
        elif self.strategy == "entropy":
            features = [entropy(all_ppl_tokens, base=2)]
        else:
            raise ValueError(f"Strategy {self.strategy} not supported")
        return features
