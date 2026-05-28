import torch


def explain_balance(values: list[float]) -> None:
    tensor = torch.tensor(values, dtype=torch.float32)
    mean = tensor.mean()
    distances = tensor - mean

    print("\nvalues:", values)
    print("mean / balance point:", round(mean.item(), 4))
    print("distances from mean:", [round(x, 4) for x in distances.tolist()])
    print("sum of distances:", round(distances.sum().item(), 4))
    print("equal-share view:", [round(mean.item(), 4)] * len(values))
    print("original total:", round(tensor.sum().item(), 4))
    print("equal-share total:", round((mean * len(values)).item(), 4))


def main() -> None:
    print("Mean as balance point and equal-share value")
    print("The balance point is where left distances and right distances cancel.")

    explain_balance([2, 4, 6])
    explain_balance([2, 4, 10])
    explain_balance([10, 10, 10, 10, 100])
    explain_balance([-2, -1, 0, 1, 2])


if __name__ == "__main__":
    main()

