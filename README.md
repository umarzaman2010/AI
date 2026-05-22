# Diffusion Models Learning Workspace

This folder is organized for learning diffusion models both theoretically and practically, from scratch.

## Start Here

Use this workspace like a small private course.

1. Read `COURSE.md` first.
2. For each module, read the matching file in `lessons/`.
3. Write your own understanding in `notes/`.
4. Implement the assigned code in `code/`.
5. Save plots and generated images in `outputs/`.
6. Record what worked, what failed, and what confused you in `learning_log.md`.

The goal is not to memorize diffusion models. The goal is to build one, debug one, and explain one.

## Learning Path

### 1. Foundations

Learn these before diffusion:

- Python scientific stack: `numpy`, `matplotlib`, `pytorch`
- Linear algebra: vectors, matrices, dot products, norms
- Probability: Gaussian distribution, expectation, variance, Bayes rule
- Calculus basics: gradients, chain rule
- Neural networks: MLPs, CNNs, loss functions, backpropagation
- PyTorch training loops: dataset, dataloader, model, optimizer, loss, checkpointing

Practice here:

- `code/00_math_basics/`
- `notes/math_and_probability.md`
- `notes/pytorch_basics.md`

### 2. Generative Modeling Basics

Before diffusion, understand what generative models try to learn:

- Data distribution vs samples
- Maximum likelihood
- Autoencoders
- VAEs
- GANs at a high level
- Score matching at a high level

Practice:

- Train a small autoencoder on MNIST or Fashion-MNIST.
- Visualize generated samples.
- Compare reconstruction vs generation.

### 3. Diffusion Theory

Core concepts:

- Forward noising process
- Reverse denoising process
- Markov chains
- Noise schedule: beta, alpha, alpha_bar
- Predicting noise instead of predicting the clean image
- DDPM objective
- Sampling loop
- Difference between training and inference

Practice here:

- `code/01_diffusion_from_scratch/`
- Implement image noising step by step.
- Plot noisy images at different timesteps.
- Train a tiny model to predict noise.

### 4. U-Net Architecture

Most image diffusion models use U-Net-like denoisers.

Learn:

- Encoder-decoder structure
- Skip connections
- ResNet blocks
- GroupNorm
- Attention blocks
- Sinusoidal timestep embeddings

Practice here:

- `code/02_unet/`
- Build a small U-Net from scratch.
- Pass random tensors through it.
- Check input and output shapes carefully.

### 5. DDPM From Scratch

Build a minimal Denoising Diffusion Probabilistic Model.

Learn:

- Training noise prediction
- MSE loss on predicted noise
- Reverse sampling formula
- Saving samples during training
- Checkpointing model weights

Practice here:

- `code/03_ddpm/`
- Train on MNIST first.
- Then try Fashion-MNIST or CIFAR-10.

### 6. Conditioning

Learn how diffusion models follow prompts, labels, or other controls.

Topics:

- Class conditioning
- Classifier-free guidance
- Text embeddings at a high level
- Cross-attention conceptually

Practice here:

- `code/04_conditioning/`
- Start with class-conditioned MNIST.
- Generate specific digits by class label.

### 7. Latent Diffusion

Modern image models often diffuse in latent space instead of pixel space.

Learn:

- Autoencoder / VAE latent space
- Why latent diffusion is faster
- Stable Diffusion architecture overview
- Text encoder, U-Net, VAE, scheduler

Practice here:

- `code/05_latent_diffusion/`
- Use a pretrained VAE later.
- Compare pixel-space vs latent-space ideas.

### 8. Experiments

Use this area for your own investigations.

Ideas:

- Compare beta schedules.
- Try fewer or more timesteps.
- Try different image sizes.
- Add attention to the U-Net.
- Compare DDPM and DDIM sampling.
- Track loss curves and generated samples.

Use:

- `code/06_experiments/`
- `configs/`
- `outputs/`
- `checkpoints/`

## Folder Guide

- `notes/` - Your own explanations, formulas, diagrams, and summaries.
- `papers/` - PDFs or links to important papers.
- `references/` - Cheatsheets, blog notes, external resources.
- `code/` - Practical implementations from simple to advanced.
- `datasets/` - Small local datasets or dataset notes.
- `outputs/` - Generated images, plots, logs.
- `checkpoints/` - Saved model weights.
- `configs/` - Experiment configuration files.

## Suggested Paper Order

1. DDPM: "Denoising Diffusion Probabilistic Models"
2. Improved DDPM: "Improved Denoising Diffusion Probabilistic Models"
3. DDIM: "Denoising Diffusion Implicit Models"
4. Score-based models: "Score-Based Generative Modeling through Stochastic Differential Equations"
5. Latent Diffusion: "High-Resolution Image Synthesis with Latent Diffusion Models"
6. Classifier-free guidance: "Classifier-Free Diffusion Guidance"

## Practical Milestones

1. Create and visualize Gaussian noise.
2. Add noise to MNIST images over timesteps.
3. Build a tiny noise-prediction neural network.
4. Build a small U-Net.
5. Train a basic DDPM on MNIST.
6. Generate samples from pure noise.
7. Add class conditioning.
8. Experiment with schedules and sampling steps.
9. Read Stable Diffusion architecture.
10. Recreate a simplified latent diffusion pipeline.

## Rule For Learning

For every formula, write code.

For every code result, make a plot.

For every plot, write a short explanation in `notes/`.

## Teacher Rule

Do not move to the next module until you can answer the checkpoint questions in the lesson file without looking.
