from typing import List, Optional

import numpy as np
from feature_extractor import FeatureExtractor
from sklearn.preprocessing import StandardScaler


class Ensemble(FeatureExtractor):
    def __init__(
        self,
        dataset,
        save_dir,
        save_embeddings_dir,
        model_variant: str = None,
        classifier: Optional[str] = None,
        protocol: Optional[str] = None,
        current_train_subset: str = "train",
        current_subset: str = "test",
        n_neighbors: int = 3,
        predict_only: bool = False,
    ):
        self.model_variant = model_variant
        self.model_variant_short = self._get_model_variant_short()
        self.model_type = "ensemble"

        super().__init__(
            dataset,
            save_dir=save_dir,
            save_embeddings_dir=save_embeddings_dir,
            classifier=classifier,
            protocol=protocol,
            current_train_subset=current_train_subset,
            current_subset=current_subset,
            n_neighbors=n_neighbors,
            predict_only=predict_only,
        )

    def load_model(self):
        raise ValueError("EnsembleFeatureExtractor does not have a model to load")

    def compute_embeddings(
        self, model, lyrics: str, logprobs=None, mp3_path=None, **kwargs
    ) -> List[List[float]]:
        raise ValueError(
            "EnsembleFeatureExtractor does not have a model to compute embeddings, use combine_embeddings instead"
        )

    def combine_embeddings(self, ensemble_model_classes):
        # predict (OOD) mode: only return test embeddings
        if self.predict_only:
            return {
                "train": [],
                "test": [model.test_preds for model in ensemble_model_classes],
            }
        # train mode: also fit scalar
        scalers = [StandardScaler() for _ in ensemble_model_classes]
        self.scalers = scalers

        ensemble_embeddings = {"train": [], "test": []}

        for i, model_class in enumerate(ensemble_model_classes):
            train_embs_scaled = scalers[i].fit_transform(model_class.train_preds)
            test_embs_scaled = scalers[i].transform(model_class.test_preds)
            # normalize by length
            if self.classifier == "knn":
                train_embs_scaled = (
                    train_embs_scaled
                    / np.sqrt(model_class.embedding_dim)
                    * model_class.weight
                )
                test_embs_scaled = (
                    test_embs_scaled
                    / np.sqrt(model_class.embedding_dim)
                    * model_class.weight
                )
            ensemble_embeddings["train"].append(train_embs_scaled)
            ensemble_embeddings["test"].append(test_embs_scaled)

        return ensemble_embeddings
