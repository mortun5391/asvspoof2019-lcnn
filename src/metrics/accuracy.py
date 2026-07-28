import torch

from src.metrics.base_metric import BaseMetric


class BinaryAccuracy(BaseMetric):
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        valid = labels >= 0
        if valid.sum() == 0:
            return 0.0
        predictions = logits[valid].argmax(dim=-1)
        return (predictions == labels[valid]).float().mean().item()
