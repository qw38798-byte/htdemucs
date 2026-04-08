from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import librosa
import numpy as np
import torch
import torch.nn as nn

try:
    from deepaudio.protocol.model.separator import Separator as _DeepAudioSeparatorBase
except Exception:
    try:
        # Keep compatibility with possible misspelled import path.
        from deepaudio.protrocol.model.separator import Separator as _DeepAudioSeparatorBase  # type: ignore
    except Exception:
        class _DeepAudioSeparatorBase(Protocol):
            pass


@dataclass
class _ArgsProxy:
    model_type: str
    start_check_point: str = ""
    lora_checkpoint: str = ""


class HTDemucsSeparatorWrapper(nn.Module, _DeepAudioSeparatorBase):  # type: ignore[misc]
    """
    HTDemucs wrapper that conforms to deepaudio.protocol.model.Separator.
    """

    def __init__(
        self,
        config_path: str = "configs/config_htdemucs_6stems.yaml",
        checkpoint_path: str = "",
        model_type: str = "htdemucs",
        device: Optional[str] = None,
        sample_rate: int = 44100,
        num_channels: int = 2,
    ) -> None:
        super().__init__()

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_channels = int(num_channels)
        self.sample_rate = int(sample_rate)
        self.sources: list[str] = []

        self._model: Optional[torch.nn.Module] = None
        self._config: Any = None

    @classmethod
    def from_version(cls, version: str) -> "HTDemucsSeparatorWrapper":
        normalized = version.strip().lower()
        if normalized in {"default", "htdemucs", "htdemucs_6s", "htdemucs-6s", "htdemucs-6stems"}:
            return cls(
                config_path="configs/config_htdemucs_6stems.yaml",
                checkpoint_path="checkpoint_path/htdemucs_6s.th",
                model_type="htdemucs",
            )
        raise ValueError(f"Unsupported HTDemucs version: {version}")

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._config is not None

    def load(self) -> "HTDemucsSeparatorWrapper":
        from inference.utils import get_model_from_config, load_start_checkpoint

        model, config = get_model_from_config(self.model_type, self.config_path)

        if self.checkpoint_path:
            args = _ArgsProxy(model_type=self.model_type, start_check_point=self.checkpoint_path)
            load_start_checkpoint(args, model, type_="inference")

        model = model.to(self.device)
        model.eval()
        self._model = model
        self._config = config
        self.num_channels = int(getattr(config.training, "channels", self.num_channels))
        self.sample_rate = int(getattr(config.training, "samplerate", self.sample_rate))
        self.sources = list(getattr(config.training, "instruments", []))
        return self

    def separate(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        *,
        force_reload: bool = False,
    ) -> Dict[str, np.ndarray]:
        from inference.utils import demix

        if force_reload or not self.loaded:
            self.load()

        assert self._model is not None
        assert self._config is not None

        wav = self._prepare_input_audio(audio)

        target_sr = int(self._config.training.samplerate)
        if sample_rate is not None and int(sample_rate) != target_sr:
            wav = np.stack(
                [
                    librosa.resample(ch, orig_sr=int(sample_rate), target_sr=target_sr)
                    for ch in wav
                ],
                axis=0,
            )

        outputs = demix(
            config=self._config,
            model=self._model,
            mix=wav,
            device=torch.device(self.device),
            model_type=self.model_type,
            pbar=False,
            use_onnx=False,
            use_compile=False,
        )

        return {stem: np.asarray(track) for stem, track in outputs.items()}

    def forward(self, mixture: torch.Tensor) -> Dict[str, torch.Tensor]:
        if mixture.ndim == 2:
            outputs = self.separate(mixture.detach().cpu().numpy(), sample_rate=self.sample_rate)
            return {k: torch.from_numpy(v).to(mixture.device, mixture.dtype) for k, v in outputs.items()}

        if mixture.ndim != 3:
            raise ValueError(f"Expected mixture shape [C,T] or [B,C,T], got {tuple(mixture.shape)}")

        batch = mixture
        if mixture.shape[1] not in (1, 2) and mixture.shape[0] in (1, 2):
            batch = mixture.permute(1, 0, 2)

        stems: Dict[str, list[torch.Tensor]] = {}
        for i in range(batch.shape[0]):
            outputs = self.separate(batch[i].detach().cpu().numpy(), sample_rate=self.sample_rate)
            for name, value in outputs.items():
                stems.setdefault(name, []).append(torch.from_numpy(value))

        return {
            name: torch.stack(values, dim=0).to(mixture.device, mixture.dtype)
            for name, values in stems.items()
        }

    @staticmethod
    def _prepare_input_audio(audio: np.ndarray) -> np.ndarray:
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)
        elif audio.ndim != 2:
            raise ValueError(f"Expected audio with ndim 1 or 2, got shape {audio.shape}")

        # Support both [T, C] and [C, T], normalize to [C, T].
        if audio.shape[0] > audio.shape[1] and audio.shape[1] in (1, 2):
            audio = audio.T

        if audio.shape[0] == 1:
            audio = np.concatenate([audio, audio], axis=0)
        elif audio.shape[0] != 2:
            raise ValueError(f"Expected mono or stereo audio, got shape {audio.shape}")

        return audio.astype(np.float32, copy=False)


HTDemucsSeparator = HTDemucsSeparatorWrapper
