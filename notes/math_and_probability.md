# Math And Probability Notes

Use this file to summarize the math you need for diffusion models.

## Must Know

- Gaussian distribution
- Mean, median, mode, variance, and standard deviation
- Standard normal noise
- Conditional probability
- Bayes rule
- Expectation
- Gradients
- Chain rule
- Matrix multiplication

## First Definitions

`mean`:

```text
the average, or center, of a group of values
```

`median`:

```text
the middle value after sorting
```

`mode`:

```text
the value that appears most often
```

`standard deviation`:

```text
how far values typically spread away from their mean
```

`variance`:

```text
standard deviation squared
```

For standard Gaussian noise:

```text
mean = 0
standard deviation = 1
variance = 1
```

If standard Gaussian noise is multiplied by `c`:

```text
new standard deviation = abs(c)
new variance = c^2
```

This explains why diffusion uses `sqrt(beta)` as a multiplier when it wants to add noise with variance `beta`.

## Diffusion-Specific Questions

- What does adding Gaussian noise mean?
- Why do we use many small noising steps?
- What is a timestep?
- What is the difference between beta, alpha, and alpha_bar?
- Why does the model predict noise?
