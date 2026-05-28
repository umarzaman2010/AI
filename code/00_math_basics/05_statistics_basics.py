from collections import Counter
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


def mode_or_none(values: list[float]) -> float | None:
    counts = Counter(values)
    most_common_value, most_common_count = counts.most_common(1)[0]
    if most_common_count == 1:
        return None
    return most_common_value


def summarize(name: str, values: torch.Tensor) -> None:
    values_list = values.tolist()
    print(f"\n{name}")
    print("-" * len(name))
    print("values:", values_list[:12], "..." if len(values_list) > 12 else "")
    print("mean:", values.mean().item())
    print("median:", median(values_list))
    print("mode:", mode_or_none(values_list))
    print("variance:", values.var(unbiased=False).item())
    print("standard deviation:", values.std(unbiased=False).item())


def main() -> None:
    output_dir = Path("outputs/statistics")
    output_dir.mkdir(parents=True, exist_ok=True)

    simple_scores = torch.tensor([50.0, 60.0, 60.0, 70.0, 90.0])
    low_spread = torch.tensor([9.0, 10.0, 11.0])
    high_spread = torch.tensor([0.0, 10.0, 20.0])

    torch.manual_seed(7)
    gaussian_noise = torch.randn(10_000)

    summarize("Simple scores", simple_scores)
    summarize("Low spread data", low_spread)
    summarize("High spread data", high_spread)
    summarize("Gaussian noise", gaussian_noise)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    axes[0].bar(range(len(simple_scores)), simple_scores.numpy())
    axes[0].set_title("Simple scores")
    axes[0].set_xlabel("item")
    axes[0].set_ylabel("value")

    axes[1].scatter(range(len(low_spread)), low_spread.numpy(), label="low spread")
    axes[1].scatter(range(len(high_spread)), high_spread.numpy(), label="high spread")
    axes[1].axhline(10, color="black", linewidth=1, linestyle="--", label="same mean")
    axes[1].set_title("Same mean, different spread")
    axes[1].legend()

    axes[2].hist(gaussian_noise.numpy(), bins=80, density=True)
    axes[2].set_title("Gaussian noise")
    axes[2].set_xlabel("value")

    fig.tight_layout()
    save_path = output_dir / "05_statistics_basics.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")


if __name__ == "__main__":
    main()

