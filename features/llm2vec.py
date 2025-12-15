from typing import List, Optional

from feature_extractor import FeatureExtractor
# from llm2vec import LLM2Vec as LLM2VecModel  # Removed self-import to avoid circular import
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoTokenizer


class LLM2VecExtractor(FeatureExtractor):
    def __init__(
        self,
        dataset,
        save_dir,
        save_embeddings_dir,
        model_variant: str = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
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
        self.model_type = "llm2vec"

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
        #     load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype="float16"
        # )

        if self.model_variant.endswith("-unsup-simcse"):
            base_model = self.model_variant.replace("-unsup-simcse", "")
        elif self.model_variant.endswith("-supervised"):
            base_model = self.model_variant.replace("-supervised", "")
        elif "McGill-NLP" not in self.model_variant:
            # XXX: for now, we only fine-tune Llama 3 MNTP models!
            base_model = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
        else:
            base_model = self.model_variant

        tokenizer = AutoTokenizer.from_pretrained(base_model)

        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)

        model = AutoModel.from_pretrained(
            base_model,
            trust_remote_code=True,
            config=config,
            # device_map={"": 0},
            # quantization_config=double_quant_config,
            torch_dtype="float16",
        )

        model = PeftModel.from_pretrained(model, self.model_variant)
        # merge lora weights
        model = model.merge_and_unload()
        model = model.to("cuda")

        # l2v = LLM2Vec.from_pretrained(
        #     base_model,
        #     peft_model_name_or_path=self.model_variant,
        #     # device_map="cuda",
        #     torch_dtype=torch.bfloat16,
        #     merge_peft=True,
        #     max_length=int(self.max_tokens),
        #     device_map={"": 0},
        # )

        l2v = LLM2VecModel(
            model, tokenizer, pooling_mode="mean", max_length=int(self.max_tokens)
        )
        return l2v

    def compute_embeddings(
        self, model, lyrics: str, logprobs=None, mp3_path=None, **kwargs
    ) -> List[float]:
        lyrics_vector = model.encode([lyrics], show_progress_bar=False)[0].tolist()
        return lyrics_vector
