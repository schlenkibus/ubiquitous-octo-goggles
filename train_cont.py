import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer



model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1',            # L1 or L2
    objective = 'pred_x0'
).cuda()

trainer = Trainer(
    diffusion,
    '/Pictures/GANfitti/generated_512_v3',
    train_batch_size = 8,
    train_lr = 1e-4,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 4,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                        # turn on mixed precision
    save_and_sample_every = 1000,
    results_folder='./results-3'
)

#trainer.load(3)

trainer.train()