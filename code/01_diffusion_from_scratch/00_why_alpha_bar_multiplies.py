import torch


def main() -> None:
    betas = torch.tensor([0.10, 0.20, 0.30])
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    print("Track only how much original signal variance remains.")
    print()
    print("step | beta  | alpha retained this step | alpha_bar retained from x_0")
    print("-----|-------|--------------------------|----------------------------")

    for step, (beta, alpha, alpha_bar) in enumerate(
        zip(betas, alphas, alpha_bars), start=1
    ):
        print(
            f"{step:>4} | {beta.item():.2f}  | {alpha.item():.2f}"
            f"{'':>20} | {alpha_bar.item():.3f}"
        )

    print()
    print("The calculation is:")
    print(f"after step 1: {alphas[0].item():.2f}")
    print(f"after step 2: {alphas[0].item():.2f} * {alphas[1].item():.2f} = {alpha_bars[1].item():.3f}")
    print(
        f"after step 3: {alphas[0].item():.2f} * {alphas[1].item():.2f} * "
        f"{alphas[2].item():.2f} = {alpha_bars[2].item():.3f}"
    )
    print()
    print("Every step keeps only a fraction of what survived before it.")


if __name__ == "__main__":
    main()

