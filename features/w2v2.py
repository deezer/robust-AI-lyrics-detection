from typing import List, Optional

import torch
import torchaudio
from feature_extractor import FeatureExtractor
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class Wav2Vec2(FeatureExtractor):
    def __init__(
        self,
        dataset,
        save_dir,
        save_embeddings_dir,
        model_variant: str = "demucs",
        hf_token: Optional[str] = None,
        classifier: Optional[str] = None,
        protocol: Optional[str] = None,
        current_train_subset: str = "train",
        current_subset: str = "test",
        n_neighbors: int = 3,
        use_demucs: bool = False,
        use_vad: bool = True,
        predict_only: bool = False,
        reencode: bool = False,
    ):
        self.model_variant = model_variant
        self.model_variant_short = self._get_model_variant_short()
        self.model_type = "w2v2"

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
            use_demucs=use_demucs,
            use_vad=use_vad,
            predict_only=predict_only,
            reencode=reencode,
        )

    def load_model(self):
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
        model = model.to("cuda")
        model = model.half()
        return processor, model

    def compute_embeddings(
        self, model, lyrics: str = None, logprobs=None, mp3_path: str = None, **kwargs
    ) -> List[List[float]]:
        processor, model = model
        if mp3_path is None:
            raise ValueError("mp3_path must be provided for W2V2 feature extraction.")
        inputs, sampling_rate = torchaudio.load(mp3_path)

        inputs = torchaudio.transforms.Resample(sampling_rate, 16_000)(inputs).mean(
            dim=0, keepdim=False
        )
        inputs = processor(inputs, sampling_rate=16_000, return_tensors="pt").to("cuda")
        inputs["input_values"] = inputs["input_values"].half()

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)[0]

        # mean pooling
        feats = torch.mean(outputs, dim=1)[0]
        feats = feats.cpu().detach().numpy().tolist()
        return feats
