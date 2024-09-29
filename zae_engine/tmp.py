from typing import Dict, Any, Optional, Union
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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

    Attributes
    ----------
    noise_scheduler : NoiseScheduler
        Noise scheduler instance.
    x_key : List[str]
        Keys representing the input data in the batch.
    """

    def __init__(self, noise_scheduler: NoiseScheduler,):
        self.noise_scheduler = noise_scheduler

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
    """
    Trainer class specialized for DDPM training and sampling.

    Inherits from the abstract Trainer class and implements the train_step and test_step methods.
    Additionally, it includes methods for generating and visualizing samples using the trained model.
    """

    def __init__(
        self,
        model,
        device: torch.device,
        mode: str,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Union[torch.optim.lr_scheduler.LRScheduler, Any]],
        noise_scheduler: NoiseScheduler,
        *,
        log_bar: bool = True,
        scheduler_step_on_batch: bool = False,
        gradient_clip: float = 0.0,
    ):
        super().__init__(
            model=model,
            device=device,
            mode=mode,
            optimizer=optimizer,
            scheduler=scheduler,
            log_bar=log_bar,
            scheduler_step_on_batch=scheduler_step_on_batch,
            gradient_clip=gradient_clip
        )
        self.noise_scheduler = noise_scheduler

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform a training step for DDPM.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch of data containing 'x_t', 'x0', 't', 'noise'.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the loss.
        """
        x_t = batch['x_t']
        t = batch['t']
        noise = batch['noise']

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
            A batch of data containing 'x_t', 'x0', 't', 'noise'.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the loss.
        """
        return self.train_step(batch=batch)

    def generate(self, batch_size: int, channels: int, height: int, width: int) -> torch.Tensor:
        """
        Generate new samples using the trained diffusion model.

        Parameters
        ----------
        batch_size : int
            Number of samples to generate.
        channels : int
            Number of channels in the generated images.
        height : int
            Height of the generated images.
        width : int
            Width of the generated images.

        Returns
        -------
        torch.Tensor
            Generated samples. Shape: (batch_size, channels, height, width)
        """
        timesteps = self.noise_scheduler.timesteps
        alpha = self.noise_scheduler.alpha
        alpha_bar = self.noise_scheduler.alpha_bar
        posterior_variance = self.noise_scheduler.posterior_variance

        # Initialize with standard normal noise
        x = torch.randn(batch_size, channels, height, width, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for t in reversed(range(timesteps)):
                t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
                # Predict the noise using the model
                noise_pred = self.model(x, t_tensor)

                if t > 0:
                    # Calculate the mean (mu_theta)
                    sqrt_recip_alpha = 1 / torch.sqrt(alpha[t])
                    sqrt_recipm_alpha = torch.sqrt(1 / alpha[t] - 1)

                    mu_theta = sqrt_recip_alpha * (x - (self.noise_scheduler.beta[t] / torch.sqrt(1 - alpha_bar[t])) * noise_pred)

                    # Sample from the posterior
                    noise = torch.randn_like(x)
                    x = mu_theta + torch.sqrt(posterior_variance[t - 1]).view(-1, 1, 1, 1) * noise
                else:
                    # For the final step, no noise is added
                    x = (x - (self.noise_scheduler.beta[t] / torch.sqrt(1 - alpha_bar[t])) * noise_pred) / torch.sqrt(alpha[t])

        return x

    def visualize_samples(self, samples: torch.Tensor, nrow: int = 4, ncol: int = 4):
        """
        Visualize generated samples.

        Parameters
        ----------
        samples : torch.Tensor
            Generated samples. Shape: (batch_size, channels, height, width)
        nrow : int, optional
            Number of rows in the grid, by default 4.
        ncol : int, optional
            Number of columns in the grid, by default 4.
        """
        # De-normalize the samples (assuming normalization was (0.5, 0.5))
        samples = samples * 0.5 + 0.5

        samples = samples.cpu().detach().numpy()
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
        for i, ax in enumerate(axes.flatten()):
            if i < samples.shape[0]:
                img = samples[i].transpose(1, 2, 0) if samples.shape[1] > 1 else samples[i].squeeze()
                cmap = None if samples.shape[1] > 1 else 'gray'
                ax.imshow(img, cmap=cmap)
            ax.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 'aux_key'를 빈 리스트로 설정 (MNIST에는 'aux' 키가 없으므로)
    collator = CollateBase(
        x_key=["x"],
        y_key=["y"],
        aux_key=[],  # No 'aux' key in MNIST
    )
    collator.add_fn(name="forward_diffusion", fn=ForwardDiffusion(noise_scheduler=NoiseScheduler()))

    # 예시 데이터셋 (MNIST 사용)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        transform=transform, 
        download=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collator.wrap()
    )

    # 모델 정의 (UNet 기반 AutoEncoder)
    model = AutoEncoder(
        block=UNetBlock, 
        ch_in=1, 
        ch_out=1, 
        width=32, 
        layers=[1, 1, 1, 1], 
        skip_connect=True
    )
    # Note: Removed `.to(device)` as Trainer handles device placement

    # 옵티마이저 및 스케줄러 정의
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingScheduler(optimizer=optimizer, T_max=50)  # T_max은 에폭 수에 맞춤

    # Trainer 인스턴스 생성
    trainer = DDPMTrainer(
        model=model,
        device=device,
        mode='train',
        optimizer=optimizer,
        scheduler=scheduler,
        log_bar=True,
        scheduler_step_on_batch=False,
        gradient_clip=0.0
    )

    # 학습 수행
    trainer.run(n_epoch=50, loader=train_loader, valid_loader=None)

    # 샘플 생성 및 시각화
    trainer.toggle("test")
    generated_samples = trainer.generate(batch_size=16, channels=1, height=28, width=28)
    trainer.visualize_samples(generated_samples, nrow=4, ncol=4)