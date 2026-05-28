def print_std_positions(mean: float, std: float) -> None:
    print(f"\nmean = {mean}, standard deviation = {std}")
    print("k std steps -> number-line position")
    print("------------------------------------")
    for k in [-3, -2, -1, 0, 1, 2, 3]:
        position = mean + k * std
        print(f"{k:+} std -> {position:+.2f}")


def main() -> None:
    print("Standard deviation is a distance from the mean.")
    print("The formula is: position = mean + k * std")

    print_std_positions(mean=0, std=2)
    print_std_positions(mean=10, std=2)
    print_std_positions(mean=0, std=0.5)


if __name__ == "__main__":
    main()

