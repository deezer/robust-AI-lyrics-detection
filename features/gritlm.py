from typing import List, Optional

from feature_extractor import FeatureExtractor
from gritlm import GritLM as GritLMModel


class GritLMExtractor(FeatureExtractor):
    def __init__(
        self,
        dataset,
        save_dir,
        save_embeddings_dir,
        model_variant: str = "GritLM/GritLM-7B",
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
        self.model_type = "gritlm"

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
        # double_quant_config = BitsAndBytesConfig(
        #     load_in_4bit=True, bnb_4bit_use_double_quant=True
        # )

        model = GritLMModel(
            self.model_variant,
            # torch_dtype="auto",
            # quantization_config=double_quant_config,
        )
        model.to("cuda")
        return model

    def compute_embeddings(
        self, model, lyrics: str, logprobs=None, mp3_path=None, **kwargs
    ) -> List[List[float]]:
        lyrics_vector = model.encode(
            [lyrics], instruction=self.gritlm_instruction("")
        ).tolist()
        return lyrics_vector

    @staticmethod
    def gritlm_instruction(instruction):
        return (
            "<|user|>\n" + instruction + "\n<|embed|>\n"
            if instruction
            else "<|embed|>\n"
        )
