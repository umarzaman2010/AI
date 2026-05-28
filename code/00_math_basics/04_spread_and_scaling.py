from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


def summarize(name: str, values: torch.Tensor) -> None:
    print(
        f"{name:<22} "
        f"mean={values.mean().item():>8.4f} "
        f"std={values.std().item():>8.4f} "
        f"variance={values.var().item():>8.4f}"
    )


def main() -> None:
    output_dir = Path("outputs/foundations")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(7)
    epsilon = torch.randn(100_000)

    scales = [0.2, 1.0, 2.0]
    scaled_noise = [scale * epsilon for scale in scales]

    print("Standard normal noise starts with mean near 0 and variance near 1.")
    print("Multiplying values scales standard deviation directly and variance by scale squared.")
    print()
    summarize("epsilon", epsilon)
    for scale, noise in zip(scales, scaled_noise):
        summarize(f"{scale} * epsilon", noise)

    fig, axes = plt.subplots(1, len(scales), figsize=(12, 3), sharey=True)

    for ax, scale, noise in zip(axes, scales, scaled_noise):
        ax.hist(noise.numpy(), bins=80, density=True)
        ax.set_title(f"{scale} * noise\nvar near {scale ** 2:.2f}")
        ax.set_xlabel("value")

    axes[0].set_ylabel("density")
    fig.tight_layout()

    save_path = output_dir / "04_spread_and_scaling.png"
    fig.savefig(save_path, dpi=150)
    print()
    print(f"Saved comparison plot to {save_path}")


if __name__ == "__main__":
    main()

