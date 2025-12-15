from typing import List, Optional

import torch
from espnet2.tasks.ssl import SSLTask
from feature_extractor import FeatureExtractor
from silero_vad import read_audio
from torch.nn.utils.rnn import pad_sequence


class XEUS(FeatureExtractor):
    def __init__(
        self,
        dataset,
        save_dir,
        save_embeddings_dir,
        model_variant: str = "xeus_checkpoint.pth",
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
        self.model_type = "xeus"

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

        model_path = self.model_variant
        xeus_model, xeus_train_args = SSLTask.build_model_from_file(
            None,
            model_path,
            "cuda:0",
        )
        xeus_model.half()
        return xeus_model

    def compute_embeddings(
        self, model, lyrics: str = None, logprobs=None, mp3_path: str = None, **kwargs
    ) -> List[List[float]]:
        if mp3_path is None:
            raise ValueError("mp3_path must be provided for XEUS feature extraction.")
        waveform = read_audio(mp3_path)

        # waveform = waveform[:500_000]

        # Convert to tensor and pad sequence
        wav_lengths = torch.LongTensor([len(waveform)]).to("cuda:0")
        waveform = (
            pad_sequence([torch.Tensor(waveform)], batch_first=True)
            .to("cuda:0")
            .squeeze(0)
            .half()
        )
        # Add dimension for batch
        waveform = waveform.unsqueeze(0)

        feats = model.encode(
            waveform, wav_lengths, use_mask=False, use_final_output=True
        )[0][-1]
        # mean pooling
        feats = torch.mean(feats, dim=1)[0]
        feats = feats.cpu().detach().numpy().tolist()
        return feats
