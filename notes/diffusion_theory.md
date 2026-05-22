# Diffusion Theory Notes

## Big Picture

A diffusion model learns to reverse a gradual noising process.

Training:

- Start with a real image.
- Pick a random timestep.
- Add known Gaussian noise.
- Ask the neural network to predict that noise.
- Minimize prediction error.

Sampling:

- Start from pure Gaussian noise.
- Repeatedly denoise step by step.
- End with a generated image.

## Key Terms

- `x_0`: original clean image
- `x_t`: noisy image at timestep `t`
- `epsilon`: Gaussian noise
- `beta_t`: amount of noise added at timestep `t`
- `alpha_t`: `1 - beta_t`
- `alpha_bar_t`: cumulative product of alphas
- `epsilon_theta`: neural network prediction of the noise

