# Model Selection Plan

The RoCoV2 analysis shows that this is a large, paired image-caption dataset with 59,958 training images, 9,904 validation images, and captions of variable length. The model-selection objective is therefore to test whether the proposed MGCN components improve caption generation, rather than comparing unrelated off-the-shelf architectures.

## Experiments

| ID | Model | Loss | Question answered |
| --- | --- | --- | --- |
| E0 | Custom single-scale attention captioner | Cross-entropy | What is the performance of a compact custom baseline? |
| E1 | MGCN encoder and visual-evidence memory decoder | Cross-entropy | Do the proposed visual and memory components help? |
| E2 | Full MGCN | Biomedical term-aware focal + coverage loss | Does the proposed objective improve difficult biomedical-word learning? |

All experiments use the same data split, vocabulary size, image resolution, caption length, optimizer, seed, and training schedule. This is essential for a fair comparison.

## Decision rule

Start with one-epoch smoke tests for E0, E1, and E2. Then train each stable model for 20 epochs. Select the final model using validation loss plus BLEU-4, METEOR, ROUGE-L, and qualitative caption inspection. The final report should include an ablation table comparing E0, E1, and E2.

## Commands

Run a smoke test:

```bash
python train_models.py --experiment baseline --data-root /path/to/rocov2 --epochs 1 --train-limit 256 --valid-limit 128
python train_models.py --experiment mgcn_ce --data-root /path/to/rocov2 --epochs 1 --train-limit 256 --valid-limit 128
python train_models.py --experiment mgcn --data-root /path/to/rocov2 --epochs 1 --train-limit 256 --valid-limit 128
```

After confirming that all runs complete, remove the sample limits and use `--epochs 20` for each experiment.
