
The purpose of this repository is to provide a simple example of the super-resolution task from the anonymous ICLR 2024 submission *Stochastic interpolants with data-dependent couplings*. 

The example is super-resolution on CIFAR10, going from 16x16 to 32x32. 

Mathematically, we let z1 = high resolution images, y = class labels, cond = low resolution images, and z0 = cond + noise. Here the pairs (z0, z1) are coupled because they are low and high resolution versions of the same image. We then train a velocity model v() to learn how to sample along the interpolant zt = az0 + bz1. We condition the model on the low resolution images by appending them to the noisy image inputs zt at each time t, i.e. dzt/dt according to the model is Model([zt, cond], t, y).

# Installation

This repository is meant to be simple and standalone. It does not require a local pip install. The main file is trainer.py which imports unet.py (for architecture) and ode_int.py (for sampling) as local modules. The main dependency of this repository is pytorch, and sampling requires the torchdiffeq library. To install torchdiffeq, run ```pip install torchdiffeq```.

# Running the example
- Get on an interactive GPU
- open trainer.py and go to the Config class
    - Change the WANDB entity and project names as needed. If not using WANDB, set use_wandb to False. In that case, please add code to save the image samples locally, which are currently not saved and only posted to WANDB.
    - Change any model hyperparameters you would like. Currently the code defaults to a 35M parameter UNet.

Then simply run:
```
python trainer.py
``` 

If using WANDB, you will see 4 concat'ed image grids. From left to right, the first grid features the low resolution images, the second grid features z0 (low resolution + noise), the third grid features model samples, and the fourth grid features high-resolution ground truth. [See an example here](https://github.com/coupledflow/couplings/blob/master/demo.png)

*Overfitting demo*: As a default, the code has an overfitting flag that is set to True. This trains the model on one batch of data and monitors the sampling for that batch. This is to be able to demo the code quickly: on CIFAR10 for 16x16 to 32x32, with the default 35M parameter model, the model samples appear visually similar to the high resolution samples in just 500-1000 gradient steps, or about 5 minutes on an a100 GPU. To run a full experiment on CIFAR10, simply switch the overfitting flag to False.


# Attribution
- The unet architecture is copied from the amazing lucidrains, [from this particular file](https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/guided_diffusion.py). Slight modifications have been made to be able to switch label conditioning on and off, and to explicitly specify the number of input and output channels in the case that they are not equal.
- The ode solver for sampling uses the wonderful [torchdiffeq library](https://github.com/rtqichen/torchdiffeq). The code to defaults to using the dopri5 adaptive solver.
- The center crop image transform is taken from the [Diffusion Transformer repository](https://github.com/facebookresearch/DiT/blob/main/train.py) who themselves take it from the [openAI guided-diffusion repository](https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126)

