# Diffusion Models From Scratch: Practical And Theoretical Course

This course is designed for one goal: understand diffusion models deeply enough that you can implement a basic DDPM yourself, then understand how modern systems like Stable Diffusion build on the same ideas.

## How To Study

Each module has four parts:

- Concept: what you must understand.
- Math: the minimum useful formulas.
- Code: the implementation task.
- Checkpoint: questions you should answer before moving on.

Recommended pace:

- Fast track: 4 weeks, 1-2 hours per day.
- Comfortable track: 8 weeks, 45-90 minutes per day.
- Deep track: 12 weeks, includes paper derivations and experiments.

## Module Map

| Module | Topic | Output |
| --- | --- | --- |
| 0 | Python, PyTorch, probability | Tensor and Gaussian noise exercises |
| 1 | Generative modeling intuition | Autoencoder baseline |
| 2 | Forward diffusion | Noising visualization |
| 3 | Noise prediction | Tiny denoising model |
| 4 | U-Net | Shape-correct U-Net denoiser |
| 5 | DDPM training | MNIST diffusion model |
| 6 | DDPM sampling | Generated images from noise |
| 7 | Conditioning | Class-conditioned digit generation |
| 8 | DDIM and faster sampling | Fewer-step sampler |
| 9 | Latent diffusion | Stable Diffusion architecture notes |
| 10 | Final project | Your own small diffusion experiment |

## Required Resources

These are the resources I recommend using. Do not jump randomly between many tutorials.

### Main Practical Course

- Hugging Face Diffusion Models Course: https://huggingface.co/learn/diffusion-course/unit0/1

Use this for guided practical learning and library-level understanding.

### Visual Intuition

- Jay Alammar, The Illustrated Stable Diffusion: https://jalammar.github.io/illustrated-stable-diffusion/
- Lilian Weng, What are Diffusion Models?: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

Use these when the math feels abstract.

### Deep Math

- Calvin Luo, Understanding Diffusion Models: A Unified Perspective: https://arxiv.org/abs/2208.11970

Use this after you understand the basic DDPM training loop.

### Core Papers

- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
- Latent Diffusion Models: https://arxiv.org/abs/2112.10752
- Classifier-Free Diffusion Guidance: https://arxiv.org/abs/2207.12598

Read papers slowly. Your first pass should answer: what problem does this solve, what is trained, what changes at sampling time?

## Weekly Plan

### Week 1: Foundations

Read:

- `lessons/00_foundations.md`
- `notes/math_and_probability.md`
- `notes/pytorch_basics.md`

Code:

- Tensor operations
- Gaussian noise generation
- Plot normal distributions
- Add noise to a simple image tensor

Deliverable:

- A notebook or script showing clean images becoming noisy.

### Week 2: Forward Diffusion

Read:

- `lessons/01_forward_diffusion.md`

Code:

- Implement beta schedule.
- Implement alpha and alpha_bar.
- Implement direct sampling of `x_t` from `x_0`.
- Visualize noising at timesteps 0, 50, 100, 300, 600, 999.

Deliverable:

- `outputs/noising_process/` with a grid of images.

### Week 3: Neural Denoising

Read:

- `lessons/02_noise_prediction.md`

Code:

- Build a tiny CNN that predicts noise.
- Train it on MNIST or synthetic 2D data.
- Compare true noise and predicted noise.

Deliverable:

- A loss curve and denoising examples.

### Week 4: U-Net

Read:

- `lessons/03_unet.md`

Code:

- Build a small U-Net.
- Add timestep embeddings.
- Confirm input and output shape are identical.

Deliverable:

- A working U-Net module with shape tests.

### Week 5: DDPM

Read:

- `lessons/04_ddpm_training.md`
- DDPM paper introduction and method sections.

Code:

- Train DDPM on MNIST.
- Save checkpoints.
- Generate samples every epoch.

Deliverable:

- First recognizable generated digits.

### Week 6: Sampling

Read:

- `lessons/05_sampling.md`

Code:

- Implement reverse sampling.
- Experiment with number of timesteps.
- Compare bad, medium, and good checkpoints.

Deliverable:

- A sample progression from noise to generated image.

### Week 7: Conditioning

Read:

- `lessons/06_conditioning.md`

Code:

- Add class label embeddings.
- Generate requested digits.
- Try classifier-free guidance.

Deliverable:

- Generated grid: rows are class labels, columns are samples.

### Week 8: Latent Diffusion And Final Project

Read:

- `lessons/07_latent_diffusion.md`
- Latent Diffusion Models paper overview.
- Jay Alammar visual guide.

Code:

- Use a pretrained pipeline only after your DDPM works.
- Write a clear architecture summary of Stable Diffusion.

Deliverable:

- Final report in `notes/final_project_report.md`.

## Final Project Options

Choose one:

- Train DDPM on Fashion-MNIST.
- Compare linear vs cosine beta schedules.
- Implement DDIM sampling.
- Build class-conditioned MNIST diffusion.
- Train on a tiny custom image folder.
- Write a visual explanation of DDPM with your own plots.

## Mastery Checklist

You understand diffusion models when you can explain:

- Why training starts with real images but sampling starts with noise.
- Why the model predicts noise.
- What `x_t`, `epsilon`, `beta`, `alpha`, and `alpha_bar` mean.
- Why the same U-Net is reused at every timestep.
- Why timestep embeddings are needed.
- Why sampling is slower than a normal forward pass.
- What classifier-free guidance changes.
- Why latent diffusion is faster than pixel diffusion.

