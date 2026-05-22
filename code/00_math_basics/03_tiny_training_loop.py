from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn


def main() -> None:
    output_dir = Path("outputs/foundations")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(7)

    x = torch.linspace(-2, 2, 200).unsqueeze(1)
    y = 2 * x + 1

    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    losses = []

    for epoch in range(100):
        prediction = model(x)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 20 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.6f}")

    learned_weight = model.weight.item()
    learned_bias = model.bias.item()

    print("Target function: y = 2x + 1")
    print(f"Learned function: y = {learned_weight:.4f}x + {learned_bias:.4f}")

    with torch.no_grad():
        prediction = model(x)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x.squeeze().numpy(), y.squeeze().numpy(), label="target")
    axes[0].plot(x.squeeze().numpy(), prediction.squeeze().numpy(), label="model")
    axes[0].set_title("Model fit")
    axes[0].legend()

    axes[1].plot(losses)
    axes[1].set_title("Training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")

    fig.tight_layout()
    save_path = output_dir / "03_tiny_training_loop.png"
    fig.savefig(save_path, dpi=150)
    print(f"Saved training plot to {save_path}")


if __name__ == "__main__":
    main()
