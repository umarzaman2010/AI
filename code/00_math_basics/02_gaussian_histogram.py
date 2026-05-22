from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


def main() -> None:
    output_dir = Path("outputs/foundations")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = torch.randn(50_000)

    mean = samples.mean().item()
    std = samples.std().item()

    print("Number of samples:", samples.numel())
    print("Mean:", mean)
    print("Standard deviation:", std)

    plt.figure(figsize=(7, 4))
    plt.hist(samples.numpy(), bins=80, density=True)
    plt.title("Samples from a standard normal distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.tight_layout()

    save_path = output_dir / "02_gaussian_histogram4.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved histogram to {save_path}")


if __name__ == "__main__":
    main()
