# SimCLR on Tiny-ImageNet-200

This project trains a SimCLR-style contrastive model to learn image embeddings on the **Tiny-ImageNet-200**.


## Installation

Create a virtual environment (recommended), then:

```bash
pip install -r requirements.txt
```

## Run
```
python hw3_simclr.py --data-root /path/to/tiny-imagenet-200 --batch-size 256 --epochs 15
```


## What the script does

- Builds a SimCLR pipeline:
  - Two augmented views per image from `train/` (random crop/flip/color jitter/blur/etc.).
  - CNN encoder + projection head to produce normalized embeddings.
  - NT-Xent contrastive loss.
- Trains until `--epochs` is reached or the loss goes below `--target-loss`.
- Produces visualizations:
  - Loss curve.
  - t-SNE thumbnail plot (first batch of `val/`).
  - Class-separation t-SNE for `--class-k` classes sampled from `train/`.
  - Nearest-neighbor retrieval for a few random queries from the first `val/` batch.

## Outputs

- Checkpoint saved to `--save` (default `simclr.pt`) containing:
  - `model` state dict.
  - Run arguments.
  - List of epoch losses.
