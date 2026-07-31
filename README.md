# ASVspoof 2019 LA LCNN

This repository contains a PyTorch-template implementation of a countermeasure
system for the Logical Access partition of ASVspoof 2019.

The project keeps the original template layout: Hydra configs live in
`src/configs`, task code is split into `datasets`, `model`, `loss`, `metrics`,
and `trainer`, and training/inference are launched through the template entry
points `train.py` and `inference.py`.

## Method

- Model: LCNN with Max-Feature-Map activations.
- Front-end: log-power STFT with a Blackman window.
- Input size: first or sampled 600 frames of an `863 x T` spectrum.
- Objective: cross-entropy over `bonafide` and `spoof`.
- Evaluation score: `logit_bonafide - logit_spoof`, so larger scores mean more
  bonafide.
- Inference: optional multi-crop scoring for eval utterances.

The LCNN architecture follows the ASVspoof 2019 STC system description and the
data preparation follows the ASVspoof 2019 LA recipe discussed in the provided
papers.

## Installation

Local CPU environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

Kaggle/Colab environment with preinstalled CUDA PyTorch:

```bash
pip install -r requirements-colab.txt
```

`requirements-colab.txt` intentionally does not install PyTorch, because Kaggle
already provides a CUDA-compatible build.

## Data

Download ASVspoof 2019 LA or attach the Kaggle dataset as notebook input. The
expected root directory is:

```text
ASVspoof2019_LA/
  ASVspoof2019_LA_train/
  ASVspoof2019_LA_dev/
  ASVspoof2019_LA_eval/
  ASVspoof2019_LA_cm_protocols/
```

On Kaggle this project was run with:

```python
DATA_ROOT = "/kaggle/input/asvpoof-2019-dataset/LA/LA"
```

The dataset files, checkpoints, logs, predictions, and the eval protocol file
are not committed to git.

## Training

Run the main LCNN recipe:

```bash
python train.py \
  -cn=asvspoof_lcnn_recipe \
  datasets.root=/path/to/ASVspoof2019_LA \
  dataloader.batch_size=8 \
  dataloader.num_workers=2 \
  writer.project_name=asvspoof2019-lcnn \
  writer.run_name=stc-lcnn-ce-v1
```

For a quick one-batch sanity check:

```bash
python train.py \
  -cn=asvspoof_one_batch \
  datasets.root=/path/to/ASVspoof2019_LA \
  writer.run_name=one-batch-check
```

Training logs and checkpoints are written by the template under:

```text
saved/<run_name>/
```

WandB logging is configured through `src/configs/writer/wandb.yaml`.

## Inference

Run multi-crop inference on the eval split:

```bash
python inference.py \
  -cn=inference_asvspoof_multicrop \
  datasets.root=/path/to/ASVspoof2019_LA \
  dataloader.batch_size=8 \
  dataloader.num_workers=2 \
  inferencer.from_pretrained=saved/stc-lcnn-ce-v1/model_best.pth \
  inferencer.save_path=predictions \
  inferencer.output_csv=submission.csv
```

The output CSV has no header and contains:

```text
utt_id,score
```

## Repository Structure

```text
train.py                         # template training entry point
inference.py                     # template inference entry point
src/configs/                     # Hydra experiment configs
src/datasets/asvspoof_dataset.py # ASVspoof LA dataset and STFT frontend
src/model/stc_lcnn.py            # LCNN model
src/loss/cross_entropy.py        # loss wrapper
src/metrics/                     # accuracy and EER
src/trainer/                     # BaseTrainer, Trainer, Inferencer
```
