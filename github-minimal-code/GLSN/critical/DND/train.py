"""Train the paper-inspired directed GAT on exact DND soft labels."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from dnd_model import DNDModel


HERE = Path(__file__).resolve().parent


def train(dataset: list, epochs: int = 50, lr: float = 3.4e-4,
          batch_size: int = 16, seed: int = 2026) -> tuple[DNDModel, list[float]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = DNDModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_nodes = 0
        for batch in loader:
            optimizer.zero_grad()
            prediction = model(batch.x, batch.edge_index)
            loss = F.binary_cross_entropy(prediction, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes
        mean_loss = total_loss / total_nodes
        losses.append(mean_loss)
        print(f"epoch {epoch + 1}/{epochs}: BCE={mean_loss:.6f}", flush=True)
    model.eval()
    return model, losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "results/train_data.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3.4e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=HERE / "results/dnd_model.pt")
    args = parser.parse_args()
    dataset = torch.load(args.data, map_location="cpu", weights_only=False)
    if not dataset:
        raise ValueError("training dataset is empty")
    model, losses = train(dataset, args.epochs, args.lr,
                          args.batch_size, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
                "num_training_graphs": len(dataset),
                "train_nodes": sorted({int(data.num_nodes) for data in dataset}),
                "feature_order": ["out_degree", "directed_betweenness",
                                  "total_degree"]}, args.out)
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "training_bce"))
        writer.writerows((i + 1, loss) for i, loss in enumerate(losses))
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
