# Biomedical Image-to-Text Analysis

This project explores a biomedical image-to-text dataset as a foundation for medical image captioning. The included notebooks inspect the RoCoV2 image-caption data, validate image files, and summarize the captions and image dimensions before model development.

## Project contents

| Path | Purpose |
| --- | --- |
| `biomedical/dataset_analysis.ipynb` | Exploratory analysis of the dataset: split counts, file types, captions, vocabulary, image dimensions, distributions, and image-integrity checks. |
| `biomedical/bio1.ipynb` | Notebook reserved for biomedical image-to-text model experiments. |

## Dataset layout

The analysis notebook expects the RoCoV2 dataset in a local directory with this structure:

```text
rocov2/
├── train_images/train/
├── valid_images/valid/
├── test_images/test/
└── train_captions.csv
```

Update `DATASET_PATH` in `biomedical/dataset_analysis.ipynb` so it points to your local copy of the dataset.

## What the analysis covers

- Counts training, validation, and testing images.
- Checks image-file extensions and identifies unreadable image files.
- Inspects the training-caption CSV for data types, missing values, duplicates, and descriptive statistics.
- Measures caption length and vocabulary frequency.
- Visualizes a randomly selected training image and the caption-length distribution.
- Estimates image dimensions from a sample of training images.

## Requirements

Use Python 3 with the following packages:

```bash
pip install jupyter pandas numpy pillow matplotlib
```

## Run the notebooks

```bash
jupyter notebook biomedical/dataset_analysis.ipynb
```

Run the cells from top to bottom after configuring `DATASET_PATH`.

## Next steps

The project is set up for the next stage: training and evaluating an image-captioning model for biomedical images. A typical workflow would include image preprocessing, caption tokenization, encoder-decoder model training, and caption-quality evaluation.

## License

No license has been specified for this repository. Add one before redistributing or reusing the code.
