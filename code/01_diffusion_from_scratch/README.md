# Lesson 01 Code: Forward Diffusion

Run these scripts from the workspace root:

```powershell
python code/01_diffusion_from_scratch/01_noise_schedule.py
python code/01_diffusion_from_scratch/02_forward_diffusion_visual.py
```

Outputs are saved in:

```text
outputs/forward_diffusion/
```

## What You Should Learn

- `beta` controls how much noise is added at each step.
- `alpha = 1 - beta`.
- `alpha_bar` is the cumulative amount of original image signal left.
- A noisy image `x_t` is a mixture of the clean image and Gaussian noise.

