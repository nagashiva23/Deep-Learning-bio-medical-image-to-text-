# Biomedical Image-to-Text

<p align="center">
  <em>Exploratory data analysis for a biomedical image-captioning workflow.</em>
</p>

This repository contains the first stage of a biomedical image-to-text project: understanding and validating the image-caption dataset before training a caption-generation model. The current implementation focuses on exploratory data analysis (EDA) of a local RoCoV2 dataset.

> **Project status:** EDA in progress. Model training and evaluation is the next phase


## Highlights

- Summarizes training, validation, and test image splits.
- Audits image file formats and detects corrupted images.
- Examines caption quality: missing values, duplicates, word counts, and vocabulary frequency.
- Visualizes sample images and caption-length distribution.
- Estimates image dimensions from a training-image sample.

## Repository structure

```text
.
└── biomedical/
    ├── dataset_analysis.ipynb  # Dataset exploration and validation
    └── bio1.ipynb              # Space for future modelling experiments
```

## Dataset

The analysis notebook expects a local RoCoV2 dataset directory with the following layout:

```text
rocov2/
├── train_images/
│   └── train/
├── valid_images/
│   └── valid/
├── test_images/
│   └── test/
└── train_captions.csv
```

The caption file should contain the training-image references and their captions. The current notebook reads the caption text from the file's second column, so preserve that column ordering or update the relevant cells to use a named column.

The dataset is not included in this repository. Obtain it separately and use it only in accordance with its licence and terms of use.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/nagashiva23/Deep-Learning-bio-medical-image-to-text-.git
cd Deep-Learning-bio-medical-image-to-text-
```

### 2. Create an environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install jupyter pandas numpy pillow matplotlib
```

### 3. Configure the dataset path

Open `biomedical/dataset_analysis.ipynb` and set `DATASET_PATH` to the absolute path of your RoCoV2 dataset. For example:

```python
from pathlib import Path

DATASET_PATH = Path("/path/to/rocov2")
```

### 4. Run the analysis

```bash
jupyter notebook biomedical/dataset_analysis.ipynb
```

Run the notebook cells from top to bottom. The first cell checks that the configured dataset directory exists.

## Analysis workflow

| Stage | What the notebook does |
| --- | --- |
| Dataset inventory | Counts images in each split and lists file extensions. |
| Caption audit | Reviews schema, missing values, duplicates, descriptive statistics, caption lengths, and frequent words. |
| Image inspection | Displays a random image and estimates image dimensions from up to 1,000 training images. |
| Data validation | Verifies that training images can be opened without errors. |

## Technology stack

- Python 3
- Jupyter Notebook
- PyTorch
- NumPy
- Pillow
- Custom Multi-Scale Gated Captioning Network (MGCN)

## Proposed training model

The project uses a custom end-to-end **Multi-Scale Gated Captioning Network (MGCN)** rather than a directly reused image-captioning architecture.

1. **Multi-Scale Gated Encoder:** three dilated convolution branches extract local, intermediate, and broad visual patterns. A learned softmax gate combines the three feature maps for each image.
2. **Visual Evidence Memory Decoder:** word generation uses spatial attention and a separate image-evidence memory, which preserves attended visual information across caption tokens.
3. **Biomedical Term-Aware Focal Loss:** a weighted focal loss increases learning signal for difficult and comparatively rare biomedical terms. An attention-coverage term discourages repeatedly focusing on one image region.

The source code is in `src/`, with the full training entry point in `train.py`.

### Train MGCN

Install the dependencies and run a small smoke test first:

```bash
python -m pip install -r requirements.txt
python train.py \
  --data-root /path/to/rocov2 \
  --epochs 1 \
  --train-limit 256 \
  --valid-limit 128
```

Then begin full training:

```bash
python train.py \
  --data-root /path/to/rocov2 \
  --epochs 20 \
  --batch-size 16 \
  --output-dir artifacts/mgcn_full
```

The trainer saves `best.pt`, `last.pt`, and `vocabulary.json` in the chosen output directory. It uses Apple Metal (MPS) when available, then CUDA, then CPU.

## Roadmap

- [x] Dataset structure inspection
- [x] Caption and image quality checks
- [ ] Image preprocessing pipeline
- [ ] Caption tokenization and vocabulary preparation
- [ ] Image-captioning model training
- [ ] Quantitative evaluation and qualitative predictions

## Contributing

Suggestions and improvements are welcome. Please open an issue describing the proposed change, or submit a focused pull request.

## License

No licence has been added yet. Please contact the repository owner before reusing or redistributing the code.
