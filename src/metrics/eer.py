import numpy as np
import torch


def compute_eer(scores, labels):
    """
    Compute Equal Error Rate for scores where larger means more bonafide.

    Labels:
        0 - bonafide
        1 - spoof
    Returns EER in percent.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    valid = labels >= 0
    scores = scores[valid]
    labels = labels[valid]

    num_bonafide = np.sum(labels == 0)
    num_spoof = np.sum(labels == 1)
    if num_bonafide == 0 or num_spoof == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    changes = np.where(np.diff(sorted_scores) != 0)[0] + 1
    starts = np.r_[0, changes, len(sorted_scores)]

    bonafide_seen = np.r_[0, np.cumsum(sorted_labels == 0)]
    spoof_seen = np.r_[0, np.cumsum(sorted_labels == 1)]

    bonafide_below = bonafide_seen[starts]
    spoof_below = spoof_seen[starts]

    frr = bonafide_below / num_bonafide
    far = (num_spoof - spoof_below) / num_spoof

    best_idx = np.argmin(np.abs(frr - far))
    return float((frr[best_idx] + far[best_idx]) * 50.0)
