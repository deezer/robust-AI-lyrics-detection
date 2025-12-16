from typing import List, Optional

import torch
from binoculars import Binoculars
from binoculars.utils import assert_tokenizer_consistency
from feature_extractor import FeatureExtractor
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class CustomBinoculars(Binoculars):
    def __init__(
        self,
        observer_name_or_path: str = "tiiuae/falcon-7b",
        performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
        use_bfloat16: bool = True,
        max_token_observed: int = 512,
        mode: str = "low-fpr",
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> None:
        DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
        DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

        print(DEVICE_1, DEVICE_2)

        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.change_mode(mode)
        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path,
            device_map={"": DEVICE_1},
            trust_remote_code=True,
            torch_dtype=(
                torch.bfloat16
                if use_bfloat16 and not quantization_config
                else torch.float32
            ),
            quantization_config=quantization_config,
        )
        self.observer_model.eval()

        print(f"Observer model loaded on {DEVICE_1}")
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path,
            device_map={"": DEVICE_2},
            trust_remote_code=True,
            torch_dtype=(
                torch.bfloat16
                if use_bfloat16 and not quantization_config
                else torch.float32
            ),
            quantization_config=quantization_config,
        )
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed


class Binoculars(FeatureExtractor):
    def __init__(
        self,
        dataset,
        save_dir,
        save_embeddings_dir,
        model_variant: str = "falcon-7b",
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
        self.model_type = "binoculars"

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
        double_quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True
        )
        # without quantization needs 2 GPUs
        # 2 GPUs + no quantization is ~4-5x faster than 1 GPU + quantization
        double_quant_config = None
        if self.model_variant == "falcon-7b":
            observer_name_or_path = "tiiuae/falcon-7b"
            performer_name_or_path = "tiiuae/falcon-7b-instruct"
        elif self.model_variant == "llama-3.1-8b":
            observer_name_or_path = "meta-llama/Llama-3.1-8B"
            performer_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
        else:
            raise ValueError(f"Invalid model variant: {self.model_variant}")

        bino = CustomBinoculars(
            max_token_observed=self.max_tokens,
            observer_name_or_path=observer_name_or_path,
            performer_name_or_path=performer_name_or_path,
            quantization_config=double_quant_config,
        )
        return bino

    def compute_embeddings(
        self, model, lyrics: str, logprobs=None, mp3_path=None, **kwargs
    ) -> List[float]:
        score = model.compute_score(lyrics)
        # Return 0.0 for invalid scores (nan, None, etc.)
        if not score:
            return [0.0]
        return [score]
