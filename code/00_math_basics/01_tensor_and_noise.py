from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


def main() -> None:
    output_dir = Path("outputs/foundations")
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_image = torch.zeros(1, 28, 28)
    noise = torch.randn_like(clean_image)
    noisy_image = clean_image + noise

    print("Clean image shape:", clean_image.shape)
    print("Noise shape:", noise.shape)
    print("Noisy image shape:", noisy_image.shape)
    print("Noise mean:", noise.mean().item())
    print("Noise std:", noise.std().item())

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    images = [clean_image, noise, noisy_image]
    titles = ["Clean image", "Gaussian noise", "Clean + noise"]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image.squeeze().numpy(), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    save_path = output_dir / "01_tensor_and_noise.png"
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    main()
