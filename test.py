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
    loss_type = 'l1',
    objective = 'pred_x0',
    beta_schedule = 'linear'
).cuda()

trainer = Trainer(
    diffusion,
    '/Pictures/GANfitti/generated_512_v3',
    train_batch_size = 8,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 8,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                        # turn on mixed precision
    save_and_sample_every = 500,
    results_folder='./results-5',
    augment_horizontal_flip = False
)

#trainer.load(8)

trainer.train()