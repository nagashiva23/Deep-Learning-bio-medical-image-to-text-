"""Run controlled model-selection experiments for RoCoV2 image captioning.

E0: custom single-scale attention baseline with cross-entropy.
E1: MGCN architecture with cross-entropy.
E2: full MGCN with biomedical term-aware focal and coverage losses.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import RoCoCaptionDataset, Vocabulary, read_captions
from src.losses import BiomedicalTermAwareFocalLoss
from src.model import MGCN, SingleScaleCaptioner


EXPERIMENTS = ("baseline", "mgcn_ce", "mgcn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one controlled image-captioning experiment.")
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-vocab", type=int, default=12000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_word_weights(vocabulary: Vocabulary, caption_count: int, enabled: bool) -> torch.Tensor:
    if not enabled:
        weights = torch.ones(len(vocabulary.id_to_token), dtype=torch.float32)
    else:
        weights = torch.tensor([
            1.0 + 0.3 * math.log((caption_count + 1) / (vocabulary.document_frequency.get(token, 0) + 1))
            for token in vocabulary.id_to_token
        ], dtype=torch.float32)
        weights /= weights.mean()
        weights = weights.clamp(max=3.0)
    weights[vocabulary.pad_id] = 0.0
    return weights


def train_epoch(model, loader, criterion, optimizer, device: torch.device, training: bool) -> float:
    model.train(training)
    total, count = 0.0, 0
    for images, captions, _ in tqdm(loader, desc="train" if training else "valid", leave=False):
        images, captions = images.to(device), captions.to(device)
        with torch.set_grad_enabled(training):
            logits, attention, _ = model(images, captions)
            loss, _ = criterion(logits, captions[:, 1:], attention)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total += loss.item() * images.size(0)
        count += images.size(0)
    return total / max(count, 1)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_root / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_captions(args.data_root / "train_captions.csv", args.train_limit)
    vocabulary = Vocabulary.build((caption for _, caption in train_rows), max_size=args.max_vocab)
    vocabulary.save(output_dir / "vocabulary.json")
    train_set = RoCoCaptionDataset(args.data_root, "train", vocabulary, args.image_size, args.max_length, args.train_limit)
    valid_set = RoCoCaptionDataset(args.data_root, "valid", vocabulary, args.image_size, args.max_length, args.valid_limit)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    full_loss = args.experiment == "mgcn"
    model = (SingleScaleCaptioner if args.experiment == "baseline" else MGCN)(len(vocabulary.id_to_token), pad_id=vocabulary.pad_id).to(device)
    criterion = BiomedicalTermAwareFocalLoss(
        make_word_weights(vocabulary, len(train_rows), enabled=full_loss),
        gamma=2.0 if full_loss else 0.0,
        pad_id=vocabulary.pad_id,
        coverage_weight=0.1 if full_loss else 0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    print(f"experiment={args.experiment} device={device} train={len(train_set)} valid={len(valid_set)} vocab={len(vocabulary.id_to_token)}")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, True)
        valid_loss = train_epoch(model, valid_loader, criterion, optimizer, device, False)
        print(f"epoch={epoch:02d} train={train_loss:.4f} valid={valid_loss:.4f}")
        checkpoint = {"experiment": args.experiment, "epoch": epoch, "model_state": model.state_dict(), "valid_loss": valid_loss, "vocab_size": len(vocabulary.id_to_token)}
        torch.save(checkpoint, output_dir / "last.pt")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
