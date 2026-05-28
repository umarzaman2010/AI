from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


def summarize(name: str, values: torch.Tensor) -> None:
    within_one_std = (values.abs() <= values.std(unbiased=False)).float().mean().item()
    print(f"\n{name}")
    print("-" * len(name))
    print("mean:", round(values.mean().item(), 4))
    print("standard deviation:", round(values.std(unbiased=False).item(), 4))
    print("smallest value:", round(values.min().item(), 4))
    print("largest value:", round(values.max().item(), 4))
    print("fraction within one standard deviation of 0:", round(within_one_std, 4))
    print("first 12 values:", [round(x, 3) for x in values[:12].tolist()])


def main() -> None:
    output_dir = Path("outputs/statistics")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(7)
    standard_noise = torch.randn(10_000)
    small_spread = 0.5 * standard_noise
    large_spread = 2.0 * standard_noise

    summarize("Noise with std about 0.5", small_spread)
    summarize("Noise with std about 2.0", large_spread)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

    axes[0].hist(small_spread.numpy(), bins=80, density=True)
    axes[0].set_title("std = 0.5\nvalues stay closer to 0")
    axes[0].set_xlim(-7, 7)

    axes[1].hist(large_spread.numpy(), bins=80, density=True)
    axes[1].set_title("std = 2.0\nvalues spread farther")
    axes[1].set_xlim(-7, 7)

    for ax in axes:
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("noise value")

    axes[0].set_ylabel("density")
    fig.tight_layout()

    save_path = output_dir / "08_standard_deviation_spread.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")


if __name__ == "__main__":
    main()

