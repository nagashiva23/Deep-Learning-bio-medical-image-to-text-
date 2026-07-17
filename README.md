# Biomedical Image-to-Text

<p align="center">
  <em>Exploratory data analysis for a biomedical image-captioning workflow.</em>
</p>

This repository contains the first stage of a biomedical image-to-text project: understanding and validating the image-caption dataset before training a caption-generation model. The current implementation focuses on exploratory data analysis (EDA) of a local RoCoV2 dataset.

> **Project status:** EDA in progress. Model training and evaluation are not yet implemented in this repository.

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
- pandas and NumPy
- Pillow
- Matplotlib

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
