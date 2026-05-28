import torch


def show_mean(name: str, values: list[float]) -> None:
    tensor = torch.tensor(values, dtype=torch.float32)
    mean = tensor.mean().item()
    deviations = tensor - mean

    print(f"\n{name}")
    print("-" * len(name))
    print("values:", values)
    print("mean:", round(mean, 4))
    print("distance from mean:", [round(x, 4) for x in deviations.tolist()])
    print("sum of distances:", round(deviations.sum().item(), 4))


def main() -> None:
    print("Mean as the balance point on a number line")
    print("The distances from the mean add up to zero.")

    show_mean("Balanced around zero", [-2, -1, 0, 1, 2])
    show_mean("Shifted right", [0, 1, 2, 3, 4])
    show_mean("Shifted left", [-4, -3, -2, -1, 0])
    show_mean("Image-like brightness values", [0.1, 0.2, 0.4, 0.7, 0.9])

    torch.manual_seed(7)
    small_noise = torch.randn(10)
    large_noise = torch.randn(10_000)

    print("\nRandom Gaussian noise")
    print("---------------------")
    print("mean of 10 samples:", round(small_noise.mean().item(), 4))
    print("mean of 10,000 samples:", round(large_noise.mean().item(), 4))
    print()
    print("The large sample is usually closer to 0 because positive and negative noise values balance better.")


if __name__ == "__main__":
    main()
