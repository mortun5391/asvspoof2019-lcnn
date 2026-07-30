import torch
from torch import nn

from src.model.mfm import MFMConv2d, MFMLinear


class STCLCNN(nn.Module):
    """
    LCNN countermeasure following the STC ASVspoof 2019 architecture.
    """

    def __init__(
        self,
        input_size=(863, 600),
        num_classes=2,
        embedding_size=80,
        dropout=0.75,
    ):
        super().__init__()
        self.features = nn.Sequential(
            MFMConv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MFMConv2d(32, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            MFMConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            MFMConv2d(48, 48, kernel_size=1, stride=1),
            nn.BatchNorm2d(48),
            MFMConv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MFMConv2d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            MFMConv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            MFMConv2d(32, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            MFMConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        flattened_size = self._infer_flattened_size(input_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            MFMLinear(flattened_size, embedding_size),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, num_classes),
        )
        self._init_weights()

    def forward(self, features, **batch):
        hidden = self.features(features)
        logits = self.classifier(hidden)
        scores = logits[:, 0] - logits[:, 1]
        return {
            "logits": logits,
            "scores": scores,
        }

    def _infer_flattened_size(self, input_size):
        freq_bins, time_frames = input_size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, freq_bins, time_frames)
            output = self.features(dummy)
        return output.reshape(1, -1).shape[1]

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
