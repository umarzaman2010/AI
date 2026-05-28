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


def main() -> None:
    output_dir = Path("outputs/forward_diffusion")
    output_dir.mkdir(parents=True, exist_ok=True)

    timesteps = 1_000
    schedule = make_linear_schedule(
        timesteps=timesteps,
        beta_start=0.0001,
        beta_end=0.02,
    )

    betas = schedule["betas"]
    alphas = schedule["alphas"]
    alpha_bars = schedule["alpha_bars"]

    print("timesteps:", timesteps)
    print("A schedule is the planned noise amount at each step.")
    print("step 1 beta:", betas[0].item())
    print("step 1000 beta:", betas[-1].item())
    print("signal retained after step 1 (alpha_bar):", alpha_bars[0].item())
    print("signal retained after step 1000 (alpha_bar):", alpha_bars[-1].item())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    axes[0].plot(betas.numpy())
    axes[0].set_title("beta")

    axes[1].plot(alphas.numpy())
    axes[1].set_title("alpha")

    axes[2].plot(alpha_bars.numpy())
    axes[2].set_title("alpha_bar")

    for ax in axes:
        ax.set_xlabel("timestep")

    fig.tight_layout()
    save_path = output_dir / "01_noise_schedule.png"
    fig.savefig(save_path, dpi=150)
    print(f"Saved schedule plot to {save_path}")


if __name__ == "__main__":
    main()
