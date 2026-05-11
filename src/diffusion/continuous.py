"""Continuous (Gaussian) diffusion wrappers. Uses diffusers DDPMScheduler."""
from diffusers import DDPMScheduler


def build_ddpm_scheduler(num_train_timesteps: int = 1000, beta_schedule: str = "linear"):
    return DDPMScheduler(num_train_timesteps=num_train_timesteps,
                         beta_schedule=beta_schedule)
