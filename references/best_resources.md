# Best Learning Resources

This file keeps you from needing to search everywhere.

## Use These In This Order

### 1. Visual Intuition

Jay Alammar, The Illustrated Stable Diffusion  
https://jalammar.github.io/illustrated-stable-diffusion/

Why use it:

- Best for understanding the full Stable Diffusion pipeline visually.
- Useful before reading the Latent Diffusion paper.

### 2. Gentle Technical Overview

Lilian Weng, What are Diffusion Models?  
https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

Why use it:

- Connects DDPM, score matching, and Langevin dynamics.
- Good bridge between intuition and papers.

### 3. Practical Course

Hugging Face Diffusion Models Course  
https://huggingface.co/learn/diffusion-course/unit0/1

Why use it:

- Practical notebooks.
- Covers Diffusers, training, conditioning, Stable Diffusion, and custom pipelines.

### 4. Deep Math

Calvin Luo, Understanding Diffusion Models: A Unified Perspective  
https://arxiv.org/abs/2208.11970

Why use it:

- Best single resource for the theory once you know the basic training loop.
- Explains the variational and score-based views.

### 5. Core Papers

DDPM  
https://arxiv.org/abs/2006.11239

DDIM  
https://arxiv.org/abs/2010.02502

Latent Diffusion Models  
https://arxiv.org/abs/2112.10752

Classifier-Free Diffusion Guidance  
https://arxiv.org/abs/2207.12598

## Do Not Start Here

Avoid starting with:

- Full Stable Diffusion codebases.
- Research repositories with many config files.
- Very high-resolution datasets.
- Prompt engineering guides.
- Video tutorials that do not make you code.

Start small. Noise a digit. Denoise a digit. Then scale.

