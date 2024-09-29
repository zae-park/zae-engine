from typing import Dict, Any
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from zae_engine.data import CollateBase
from zae_engine.models import AutoEncoder
from zae_engine.schedulers import CosineAnnealingScheduler
from zae_engine.nn_night.blocks import UNetBlock
from zae_engine.trainer import Trainer


class NoiseScheduler:
    """
    Scheduler for managing the noise levels in DDPM.

    Parameters
    ----------
    timesteps : int
        Total number of diffusion steps.
    schedule : str, optional
        Type of noise schedule ('linear', 'cosine'). Default is 'linear'.
    beta_start : float, optional
        Starting value of beta. Default is 1e-4.
    beta_end : float, optional
        Ending value of beta. Default is 0.02.

    Attributes
    ----------
    beta : torch.Tensor
        Noise levels for each timestep.
    alpha : torch.Tensor
        1 - beta for each timestep.
    alpha_bar : torch.Tensor
        Cumulative product of alpha up to each timestep.
    sqrt_alpha_bar : torch.Tensor
        Square root of alpha_bar.
    sqrt_one_minus_alpha_bar : torch.Tensor
        Square root of (1 - alpha_bar).
    posterior_variance : torch.Tensor
        Variance used in the posterior distribution.
    """

    def __init__(self, timesteps=1000, schedule='linear', beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps

        if schedule == 'linear':
            self.beta = self.linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == 'cosine':
            self.beta = self.cosine_beta_schedule(timesteps)
        else:
            raise NotImplementedError(f"Schedule '{schedule}' is not implemented.")

        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # Precompute terms for efficiency
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        self.posterior_variance = self.beta * (1 - self.alpha_bar[:-1]) / (1 - self.alpha_bar[1:])

    def linear_beta_schedule(self, timesteps, beta_start, beta_end):
        """
        Linear schedule for beta.

        Parameters
        ----------
        timesteps : int
            Total number of diffusion steps.
        beta_start : float
            Starting value of beta.
        beta_end : float
            Ending value of beta.

        Returns
        -------
        torch.Tensor
            Beta schedule.
        """
        return torch.linspace(beta_start, beta_end, timesteps)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule for beta as proposed in https://arxiv.org/abs/2102.09672.

        Parameters
        ----------
        timesteps : int
            Total number of diffusion steps.
        s : float, optional
            Small offset to prevent beta from being exactly 0. Default is 0.008.

        Returns
        -------
        torch.Tensor
            Beta schedule.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas_cumprod = alphas_cumprod[:timesteps]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, min=1e-4, max=0.999)
        return betas


class ForwardDiffusion:
    """
    Class for performing the forward diffusion process by adding noise to the input data.

    Parameters
    ----------
    noise_scheduler : NoiseScheduler
        Instance of NoiseScheduler managing the noise levels.
    x_key : List[str]
        The key in the batch dictionary that represents the input data.

    Attributes
    ----------
    noise_scheduler : NoiseScheduler
        Noise scheduler instance.
    x_key : List[str]
        Keys representing the input data in the batch.
    """

    def __init__(self, noise_scheduler: NoiseScheduler, x_key: list = ["x"], device: str = 'cuda'):
        self.noise_scheduler = noise_scheduler
        self.x_key = x_key

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies forward diffusion by adding noise to the input data at a random timestep.

        Parameters
        ----------
        batch : Dict[str, Any]
            The input batch containing data under keys specified in x_key.

        Returns
        -------
        Dict[str, Any]
            The batch with added noise, original data, and timestep information.
            Contains:
                - 'x_t': Noised data.
                - 'x0': Original data.
                - 't': Timestep.
                - 'noise': Added noise.
        """
        # Initialize dictionaries to hold results
        batch_noised = {}
        batch_original = {}
        batch_timestep = {}
        batch_noise = {}

        key = "x"
        origin = batch[key]
        if key not in batch:
            raise KeyError(f"Batch must contain '{key}' key.")

        # Sample random timesteps for each sample in the batch
        t = torch.randint(0, self.noise_scheduler.timesteps, (1, )).long()
        noise = torch.randn_like(origin)
        batch["t"] = t
        batch["noise"] = noise

        # Calculate x_t
        sqrt_alpha_bar_t = self.noise_scheduler.sqrt_alpha_bar[t].view(1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.noise_scheduler.sqrt_one_minus_alpha_bar[t].view(1, 1, 1)
        x_t = sqrt_alpha_bar_t * origin + sqrt_one_minus_alpha_bar_t * noise
        batch["x_t"]  = x_t

        return batch


class DDPMTrainer(Trainer):
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform a training step for DDPM.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch of data containing 'x_t', 'x0', 't_x', 'noise_x'.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the loss.
        """
        x_t = batch['x_t']
        t = batch['t_x']
        noise = batch['noise_x']

        # 모델은 x_t과 t를 입력으로 받아 노이즈를 예측
        noise_pred = self.model(x_t, t)

        # 손실 계산 (예: MSE)
        loss = nn.MSELoss()(noise_pred, noise)

        return {"loss": loss}

    def test_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform a testing step for DDPM.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch of data containing 'x_t', 'x0', 't_x', 'noise_x'.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the loss.
        """
        x_t = batch['x_t']
        t = batch['t_x']
        noise = batch['noise_x']

        # 모델은 x_t과 t를 입력으로 받아 노이즈를 예측
        noise_pred = self.model(x_t, t)

        # 손실 계산 (예: MSE)
        loss = nn.MSELoss()(noise_pred, noise)

        return {"loss": loss}


if __name__ == "__main__":
    # ForwardDiffusion 인스턴스 생성
    collator = CollateBase()
    collator.add_fn(
        name="forward_diffusion", 
        fn=ForwardDiffusion(noise_scheduler=NoiseScheduler())
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    

    dataloader = DataLoader(dataset, batch_size=128,shuffle=True, collate_fn=collator.wrap())
    model = AutoEncoder(block=UNetBlock, ch_in=3, ch_out=1, width=32, layers=[1, 1, 1, 1], skip_connect=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingScheduler(optimizer=optimizer)
    
    # Trainer 인스턴스 생성
    trainer = DDPMTrainer(
        model=model,
        device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
        optimizer=optimizer, 
        scheduler=scheduler
    )

    # 학습 수행
    trainer.run(n_epoch=50, loader=dataloader)

    # 샘플 생성 및 시각화
    trainer.toggle("test")
    generated_samples = trainer.generate(batch_size=16, channels=1, height=28, width=28)
    trainer.visualize_samples(generated_samples, nrow=4, ncol=4)