import math
import os
from typing import Dict, Any, Optional, Union, Type, Sequence, Callable
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from zae_engine.data import CollateBase
from zae_engine.models import AutoEncoder
from zae_engine.models.converter import DimConverter
from zae_engine.schedulers import CosineAnnealingScheduler
from zae_engine.nn_night.blocks import UNetBlock
from zae_engine.trainer import Trainer


class CustomMNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.mnist = datasets.MNIST(root=root, train=train, transform=transform, download=download)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        return {"pixel_values": image}


class TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(TimestepEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, embed_dim)
        print(f"TimestepEmbedding initialized with embed_dim={self.embed_dim}")

    def forward(self, t):
        """
        Sinusoidal embedding for timesteps.
        """
        if t.dim() > 1:
            t = t.view(-1)  # Ensure t is (batch_size,)
            print(f"TimestepEmbedding forward: Reshaped t shape: {t.shape}")

        half_dim = self.embed_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]  # (batch_size, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (batch_size, embed_dim)

        # Debugging prints
        # print(f"TimestepEmbedding input t shape: {t.shape}")  # Should be (batch_size,)
        # print(f"Sin/Cos embedding shape: {emb.shape}")  # Should be (batch_size, embed_dim)

        emb = self.linear(emb)  # (batch_size, embed_dim)

        # print(f"TimestepEmbedding output emb shape: {emb.shape}")  # Should be (batch_size, embed_dim)

        return emb


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

    def __init__(self, timesteps=1000, schedule="linear", beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps

        if schedule == "linear":
            self.beta = self.linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            self.beta = self.cosine_beta_schedule(timesteps)
        else:
            raise NotImplementedError(f"Schedule '{schedule}' is not implemented.")

        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # Precompute terms for efficiency
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        self.posterior_variance = self.beta[1:] * (1 - self.alpha_bar[:-1]) / (1 - self.alpha_bar[1:])

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
    """

    def __init__(self, noise_scheduler: NoiseScheduler):
        self.noise_scheduler = noise_scheduler

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies forward diffusion by adding noise to the input data at a random timestep.

        Parameters
        ----------
        batch : Dict[str, Any]
            The input batch containing data under 'pixel_values'.

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
        key = "pixel_values"
        if key not in batch:
            raise KeyError(f"Batch must contain '{key}' key.")

        origin = batch[key]
        # origin has shape [channel, height, width]

        # Sample random timestep
        t = torch.randint(0, self.noise_scheduler.timesteps, (1,)).long()

        noise = torch.randn_like(origin)

        # Calculate x_t
        sqrt_alpha_bar_t = self.noise_scheduler.sqrt_alpha_bar[t].view(1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.noise_scheduler.sqrt_one_minus_alpha_bar[t].view(1, 1, 1)
        x_t = sqrt_alpha_bar_t * origin + sqrt_one_minus_alpha_bar_t * noise

        batch["x_t"] = x_t
        batch["x0"] = origin
        batch["t"] = t
        batch["noise"] = noise

        return batch


class DDPM(AutoEncoder):

    def __init__(
        self,
        block: Type[Union[UNetBlock, nn.Module]],
        ch_in: int,
        ch_out: int,
        width: int,
        layers: Sequence[int],
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        skip_connect: bool = False,
        timestep_embed_dim: int = 256,  # 타임스탬프 임베딩 차원
    ):
        super(DDPM, self).__init__(block, ch_in, ch_out, width, layers, groups, dilation, norm_layer, skip_connect)

        # 타임스탬프 임베딩 모듈 추가
        self.timestep_embedding = TimestepEmbedding(timestep_embed_dim)
        # 타임스탬프 임베딩을 추가하기 위한 추가 레이어
        self.t_embed_proj = nn.Linear(timestep_embed_dim, width * 16)
        # print(f"DDPM initialized with timestep_embed_dim={timestep_embed_dim}")

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: 노이즈가 추가된 입력 이미지 (x_t) - Shape: (batch_size, channels, height, width)
        t: 타임스탬프 (timestep) - Shape: (batch_size,)
        """
        # Ensure t is (batch_size,)
        if t.dim() > 1:
            t = t.view(-1)

        self.feature_vectors = []

        # Forwarding encoder & hook immediate outputs
        _ = self.encoder(x)
        if not self.feature_vectors:
            raise ValueError("No feature vectors collected from encoder.")
        feat = self.bottleneck(self.feature_vectors.pop())
        self.feature_vectors = []

        # 타임스탬프 임베딩
        t_emb = self.timestep_embedding(t)  # Shape: (batch_size, embed_dim)
        t_emb = self.t_embed_proj(t_emb)  # Shape: (batch_size, width * 16)
        t_emb = t_emb[:, :, None, None]  # Shape: (batch_size, width * 16, 1, 1)
        feat = feat + t_emb  # Broadcasting addition

        # Decoder with skip connections if enabled
        for up_pool, dec in zip(self.up_pools, self.decoder):
            feat = up_pool(feat)
            if self.skip_connect and len(self.feature_vectors) > 0:
                feat = torch.cat((feat, self.feature_vectors.pop()), dim=1)
            feat = dec(feat)

        output = self.sig(self.fc(feat))
        return output


class DDPMTrainer(Trainer):
    """
    Trainer class specialized for DDPM training and sampling.

    Inherits from the abstract Trainer class and implements the train_step and test_step methods.
    Additionally, it includes methods for generating and visualizing samples using the trained model.
    """

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
        x_t = batch["x_t"]
        t = batch["t"]
        noise = batch["noise"]

        # Ensure t is (batch_size,)
        if t.dim() > 1:
            t = t.view(-1)

        # 모델은 x_t과 t를 입력으로 받아 노이즈를 예측
        noise_pred = self.model(x_t, t)

        # 손실 계산 (예: MSE)
        loss = nn.MSELoss()(noise_pred, noise)

        return {"loss": loss}

    def test_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.train_step(batch=batch)

    def noise_scheduling(self, noise_scheduler):
        self.noise_scheduler = noise_scheduler

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
        x = self._to_device(torch.randn(batch_size, channels, height, width))
        print(f"Generated noise x shape: {x.shape}")

        self.model.eval()
        with torch.no_grad():
            for t_step in reversed(range(timesteps)):
                t_tensor = self._to_device(torch.full((batch_size,), t_step, dtype=torch.long))
                # Predict the noise using the model
                noise_pred = self.model(x, t_tensor)
                print(f"At timestep {t_step}, noise_pred shape: {noise_pred.shape}")

                if t_step > 0:
                    # Calculate the mean (mu_theta)
                    sqrt_recip_alpha = 1 / torch.sqrt(alpha[t_step])
                    sqrt_recipm_alpha = torch.sqrt(1 / alpha[t_step] - 1)

                    mu_theta = sqrt_recip_alpha * (
                        x - (self.noise_scheduler.beta[t_step] / torch.sqrt(1 - alpha_bar[t_step])) * noise_pred
                    )

                    # Sample from the posterior
                    noise = torch.randn_like(x)
                    var = self._to_device(torch.sqrt(posterior_variance[t_step - 1]).view(-1, 1, 1, 1))
                    x = mu_theta + var * noise
                else:
                    # For the final step, no noise is added
                    x = (
                        x - (self.noise_scheduler.beta[t_step] / torch.sqrt(1 - alpha_bar[t_step])) * noise_pred
                    ) / torch.sqrt(alpha[t_step])

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
        samples = samples.cpu().detach().numpy()
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
        for i, ax in enumerate(axes.flatten()):
            if i < samples.shape[0]:
                img = samples[i].transpose(1, 2, 0) if samples.shape[1] > 1 else samples[i].squeeze()
                cmap = None if samples.shape[1] > 1 else "gray"
                ax.imshow(img, cmap=cmap)
            ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    epoch = 3
    target_width = target_height = 64
    data_path = "./mnist_example"

    # NoiseScheduler 인스턴스 생성
    noise_scheduler = NoiseScheduler()
    print("NoiseScheduler initialized.")

    # ForwardDiffusion 인스턴스 생성
    forward_diffusion = ForwardDiffusion(noise_scheduler=noise_scheduler)
    print("ForwardDiffusion initialized.")

    # 예시 데이터셋 (MNIST 사용)
    transform = transforms.Compose(
        [transforms.Resize((target_height, target_width)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = CustomMNISTDataset(root=data_path, train=True, transform=transform, download=True)
    print("CustomMNISTDataset initialized.")

    # 'aux_key'를 빈 리스트로 설정 (MNIST에는 'aux' 키가 없으므로)
    collator = CollateBase(x_key=["pixel_values", "t", "x_t", "noise"], y_key=[], aux_key=[])
    collator.add_fn(name="forward_diffusion", fn=forward_diffusion)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator.wrap())
    print("DataLoader initialized.")

    # 모델 정의 (UNet 기반 AutoEncoder)
    model = DDPM(
        block=UNetBlock, ch_in=1, ch_out=1, width=8, layers=[1, 1, 1, 1], skip_connect=False, timestep_embed_dim=256
    )
    print("Model defined and moved to device.")

    # 옵티마이저 및 스케줄러 정의
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingScheduler(optimizer=optimizer, total_iters=50)  # total_iters를 에폭 수에 맞춤
    print("Optimizer and scheduler initialized.")

    # Trainer 인스턴스 생성
    trainer = DDPMTrainer(
        model=model,
        device=device,
        mode="train",
        optimizer=optimizer,
        scheduler=scheduler,
        log_bar=True,
        scheduler_step_on_batch=False,
        gradient_clip=0.0,
    )
    trainer.noise_scheduling(noise_scheduler)
    print("Trainer initialized.")

    # 학습 수행
    print("Starting training...")
    trainer.run(n_epoch=epoch, loader=train_loader, valid_loader=None)
    print("Training completed.")

    # 샘플 생성 및 시각화
    print("Generating samples...")
    trainer.toggle("test")
    generated_samples = trainer.generate(batch_size=16, channels=1, height=target_height, width=target_width)
    trainer.visualize_samples(generated_samples, nrow=4, ncol=4)
    print("Sample generation and visualization completed.")

    os.remove(data_path)
