import random
from typing import List

import librosa
import numpy as np
from pedalboard import Gain, HighpassFilter, LowpassFilter, Pedalboard, PitchShift, Reverb

default_adversarial_params = {
    "adversarial_pitch": 2,
    "adversarial_stretch": 0.2,  # -> [80%, 120%]
    "adversarial_noise_std": 0.01,
}

ALL_EFFECTS = ["pitch", "reverb", "eq", "noise", "stretch"]


class AdversarialAugmenter:
    def __init__(self, selected_effects: List[str], params={}):
        self.effects = selected_effects
        if any([effect not in ALL_EFFECTS for effect in self.effects]):
            raise ValueError(
                f"Invalid effect selected: {self.effects}. Choose from {ALL_EFFECTS}"
            )

        self.params = default_adversarial_params
        self.params.update(params)
        self.sample_rate = 16000

    def create_augmentation(self, audio):
        board = Pedalboard()

        for effect in self.effects:
            if effect == "pitch":
                sign = 2 * np.random.randint(0, 2) - 1
                fact = np.random.randint(1, self.params["adversarial_pitch"] + 1)
                semitones = fact * sign
                board.append(PitchShift(semitones=semitones))

            if effect == "reverb":
                board.append(
                    Reverb(
                        room_size=np.random.uniform(0.2, 0.8),
                        damping=np.random.uniform(0.2, 0.8),
                        wet_level=np.random.uniform(0.2, 0.8),
                        dry_level=np.random.uniform(0.2, 0.8),
                        width=np.random.choice([1, np.random.uniform(0.5, 1)]),
                    )
                )
            if effect == "eq":
                # Use HighpassFilter and LowpassFilter for a more controlled EQ effect
                if not random.randrange(3):  # Simulate Bandreject
                    center_freq = np.random.uniform(
                        500, 4000
                    )  # center around 500-4000 Hz
                    band_width = np.random.uniform(
                        50, 500
                    )  # bandwidth between 50-500 Hz

                    highpass_cutoff = center_freq - band_width / 2
                    lowpass_cutoff = center_freq + band_width / 2

                    board.append(HighpassFilter(cutoff_frequency_hz=highpass_cutoff))
                    board.append(LowpassFilter(cutoff_frequency_hz=lowpass_cutoff))
                    board.append(Gain(gain_db=np.random.uniform(-12, -3)))  # gain cut

                if not random.randrange(3):  # Simulate Bass Boost/Cut
                    lowpass_cutoff = np.random.uniform(80, 250)
                    board.append(LowpassFilter(cutoff_frequency_hz=lowpass_cutoff))
                    board.append(Gain(gain_db=np.random.uniform(-12, 12)))

                if not random.randrange(3):  # Simulate Treble Boost/Cut
                    highpass_cutoff = np.random.uniform(2000, 8000)
                    board.append(HighpassFilter(cutoff_frequency_hz=highpass_cutoff))
                    board.append(Gain(gain_db=np.random.uniform(-12, 12)))

            if "noise" in self.effects:
                if np.random.rand() > 0.5:  # Apply noise in 50% of cases
                    noise = np.random.normal(
                        0, self.params["adversarial_noise_std"], len(audio)
                    )
                    audio += noise

            if "stretch" in self.effects:
                stretch_factor = np.random.uniform(
                    1 - self.params["adversarial_stretch"],
                    1 + self.params["adversarial_stretch"],
                )
                audio = librosa.effects.time_stretch(audio, rate=stretch_factor)

        return board, audio

    def py_apply_augmentation(self, audio):
        # Pedalboard expects a NumPy array
        audio_np = audio.numpy()

        board, audio_np = self.create_augmentation(audio_np)

        # Apply the pedalboard
        effected = board(audio_np, self.sample_rate)

        effected = effected / np.max(np.abs(effected))

        return effected
