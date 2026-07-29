from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class ASVspoofDataset(BaseDataset):
    """
    Dataset for the ASVspoof 2019 Logical Access countermeasure task.

    Labels follow a single convention across the project:
        bonafide -> 0
        spoof    -> 1
    """

    PROTOCOL_NAMES = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }

    AUDIO_DIRS = {
        "train": "ASVspoof2019_LA_train",
        "dev": "ASVspoof2019_LA_dev",
        "eval": "ASVspoof2019_LA_eval",
    }

    LABELS = {
        "bonafide": 0,
        "spoof": 1,
    }

    def __init__(
        self,
        root,
        part,
        protocol_path=None,
        sample_rate=16000,
        n_fft=1724,
        win_length=1724,
        hop_length=130,
        target_frames=600,
        random_crop=False,
        center_crop=True,
        max_items_per_label=None,
        eps=1e-10,
        *args,
        **kwargs,
    ):
        self.root = self._resolve_path(root)
        self.part = part
        self.protocol_path = (
            self._resolve_path(protocol_path)
            if protocol_path is not None
            else self._default_protocol_path()
        )
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.target_frames = target_frames
        self.random_crop = random_crop
        self.center_crop = center_crop
        self.max_items_per_label = max_items_per_label
        self.eps = eps

        self.window = torch.blackman_window(self.win_length)
        index = self._build_index()
        if self.max_items_per_label is not None:
            index = self._limit_items_per_label(index)
        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        item = self._index[ind]
        waveform = self.load_object(item["path"])
        features = self._waveform_to_features(waveform)

        return {
            "features": features,
            "labels": item["label"],
            "utt_id": item["utt_id"],
        }

    def load_object(self, path):
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Unexpected sample rate for {path}: "
                f"{sample_rate}, expected {self.sample_rate}"
            )

        waveform = torch.from_numpy(waveform)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)
        return waveform

    def _build_index(self):
        if self.part not in self.AUDIO_DIRS:
            raise ValueError(f"Unknown ASVspoof partition: {self.part}")
        if not self.protocol_path.exists():
            raise FileNotFoundError(f"Protocol file not found: {self.protocol_path}")

        audio_dir = self.root / self.AUDIO_DIRS[self.part] / "flac"
        if not audio_dir.exists():
            audio_dir = self.root / self.AUDIO_DIRS[self.part]
        index = []
        with self.protocol_path.open("r", encoding="utf-8") as protocol:
            for line in protocol:
                fields = line.strip().split()
                if len(fields) < 2:
                    continue

                speaker_id = fields[0]
                utt_id = fields[1]
                attack_id = fields[3] if len(fields) > 3 else "-"
                label = self._parse_label(fields[-1])
                path = audio_dir / f"{utt_id}.flac"

                index.append(
                    {
                        "path": str(path),
                        "label": label,
                        "utt_id": utt_id,
                        "speaker_id": speaker_id,
                        "attack_id": attack_id,
                    }
                )

        return index

    def _default_protocol_path(self):
        return (
            self.root / "ASVspoof2019_LA_cm_protocols" / self.PROTOCOL_NAMES[self.part]
        )

    def _parse_label(self, value):
        return self.LABELS.get(value, -1)

    def _limit_items_per_label(self, index):
        counts = {}
        limited_index = []
        for item in index:
            label = item["label"]
            if label < 0:
                continue
            count = counts.get(label, 0)
            if count >= self.max_items_per_label:
                continue
            limited_index.append(item)
            counts[label] = count + 1
        return limited_index

    def _resolve_path(self, path):
        path = Path(path).expanduser()
        if path.is_absolute():
            return path
        return ROOT_PATH / path

    def _waveform_to_features(self, waveform):
        if waveform.numel() < self.win_length:
            waveform = F.pad(waveform, (0, self.win_length - waveform.numel()))

        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        log_power = torch.log(spectrum.abs().pow(2) + self.eps)
        log_power = self._fix_num_frames(log_power)
        return log_power.unsqueeze(0)

    def _fix_num_frames(self, features):
        num_frames = features.shape[-1]
        if num_frames < self.target_frames:
            return F.pad(features, (0, self.target_frames - num_frames))
        if num_frames == self.target_frames:
            return features

        max_start = num_frames - self.target_frames
        if self.random_crop:
            start = torch.randint(0, max_start + 1, size=(1,)).item()
        elif self.center_crop:
            start = max_start // 2
        else:
            start = 0
        return features[:, start : start + self.target_frames]
