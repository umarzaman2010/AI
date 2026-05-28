from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


def make_linear_schedule(timesteps: int, beta_start: float, beta_end: float) -> dict[str, torch.Tensor]:
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
    }


def make_simple_image(size: int = 28) -> torch.Tensor:
    image = torch.zeros(1, size, size)

    image[:, 6:22, 12:16] = 1.0
    image[:, 12:16, 6:22] = 1.0

    return image


def q_sample(x_0: torch.Tensor, step: int, alpha_bars: torch.Tensor) -> torch.Tensor:
    if step == 0:
        return x_0

    noise = torch.randn_like(x_0)
    alpha_bar_t = alpha_bars[step - 1]

    clean_weight = torch.sqrt(alpha_bar_t)
    noise_weight = torch.sqrt(1.0 - alpha_bar_t)

    return clean_weight * x_0 + noise_weight * noise


def main() -> None:
    output_dir = Path("outputs/forward_diffusion")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(7)

    timesteps = 1_000
    schedule = make_linear_schedule(
        timesteps=timesteps,
        beta_start=0.0001,
        beta_end=0.02,
    )

    alpha_bars = schedule["alpha_bars"]
    x_0 = make_simple_image()

    chosen_steps = [0, 1, 50, 100, 300, 600, 1000]

    fig, axes = plt.subplots(1, len(chosen_steps), figsize=(14, 2.4))

    for ax, step in zip(axes, chosen_steps):
        x_t = q_sample(x_0, step, alpha_bars)
        ax.imshow(x_t.squeeze().numpy(), cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"step={step}")
        ax.axis("off")

        if step == 0:
            print("step=000 clean starting image: clean_weight=1.000000 noise_weight=0.000000")
        else:
            alpha_bar_t = alpha_bars[step - 1]
            print(
                f"step={step:04d} "
                f"alpha_bar={alpha_bar_t.item():.6f} "
                f"clean_weight={torch.sqrt(alpha_bar_t).item():.6f} "
                f"noise_weight={torch.sqrt(1.0 - alpha_bar_t).item():.6f}"
            )

    fig.tight_layout()
    save_path = output_dir / "02_forward_diffusion_visual.png"
    fig.savefig(save_path, dpi=150)
    print(f"Saved forward diffusion visual to {save_path}")


if __name__ == "__main__":
    main()
