"""Save a small example model to a .pt file for testing checkpoint loading.

Usage:
    python examples/save_example_model.py

Creates examples/example_model.pt containing a small MLP.
"""

import torch
import torch.nn as nn
from pathlib import Path


def main():
    # Small MLP: 784 -> 128 -> 64 -> 10
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    out_path = Path(__file__).parent / "example_model.pt"
    torch.save(model, out_path)
    print(f"Saved example model to {out_path}")
    print(f"  Architecture: MLP(784 -> 128 -> 64 -> 10)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    print("Analyze with:")
    print(f"  python -m src.cli diagnose --checkpoint {out_path}")


if __name__ == "__main__":
    main()
