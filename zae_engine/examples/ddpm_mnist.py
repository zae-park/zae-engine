import math
import os.path
import shutil
from typing import Dict, Any, Union, Type, Sequence, Callable, Tuple, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from zae_engine.data import CollateBase
from zae_engine.models import AutoEncoder
from zae_engine.schedulers import CosineAnnealingScheduler, WarmUpScheduler, SchedulerChain
from zae_engine.nn_night.blocks import UNetBlock
from zae_engine.trainer import Trainer


class CustomMNISTDataset(Dataset):
    """
    Custom Dataset class for MNIST data.

    This class wraps the torchvision MNIST dataset and returns a dictionary containing the image.

    Parameters
    ----------
    root : str
        Root directory of dataset where MNIST exists or will be saved.
    train : bool, optional
        If True, creates dataset from training set, otherwise from test set.
    transform : callable, optional
        A function/transform that takes in an image and returns a transformed version.
    download : bool, optional
        If True, downloads the dataset from the internet and puts it in root directory.
    """

    def __init__(self, root, train=True, transform=None, download=False):
        self.mnist = datasets.MNIST(root=root, train=train, transform=transform, download=download)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        return {"pixel_values": image}


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal embedding for timesteps.

    This module generates sinusoidal embeddings for each timestep, similar to positional encodings used in Transformers.

    Parameters
    ----------
    embed_dim : int
        Dimension of the embedding vector.
    """

    def __init__(self, embed_dim):
        super(TimestepEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, t: Tensor) -> Tensor:
        """
        Forward pass for timestep embedding.

        Parameters
        ----------
        t : Tensor
            Timestep tensor of shape (batch_size,) or higher.

        Returns
        -------
        Tensor
            Embedded timestep tensor of shape (batch_size, embed_dim).
        """
        if t.dim() > 1:
            t = t.view(-1)  # Ensure t is (batch_size,)
            print(f"TimestepEmbedding forward: Reshaped t shape: {t.shape}")

        half_dim = self.embed_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]  # (batch_size, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (batch_size, embed_dim)

        emb = self.linear(emb)  # (batch_size, embed_dim)

        return emb


class NoiseScheduler:
    """
    Scheduler for managing the noise levels in DDPM.

    This class defines the noise schedule used in the diffusion process, supporting both linear and cosine schedules.

    Parameters
    ----------
    timesteps : int, optional
        Total number of diffusion steps. Default is 1000.
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
    timesteps : int
        Total number of diffusion steps.
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

    def linear_beta_schedule(self, timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
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

    def cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
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

    def get_sigma(self, t: torch.Tensor, ddim: bool = False) -> torch.Tensor:
        """
        Get the sigma value for a given timestep.

        Parameters
        ----------
        t : torch.Tensor
            Timestep tensor.
        ddim : bool, optional
            Whether to use DDIM sampling. If True, uses sigma=0 for deterministic sampling.

        Returns
        -------
        torch.Tensor
            Sigma value for the given timestep.
        """
        if ddim:
            return torch.zeros_like(self.beta[t])  # Deterministic sampling for DDIM
        else:
            return torch.sqrt(self.posterior_variance[t])


class ForwardDiffusion:
    """
    Class for performing the forward diffusion process by adding noise to the input data.

    This class adds noise to the input data at a randomly sampled timestep, producing the noised data `x_t`,
    along with the original data `x0`, the timestep `t`, and the noise added.

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
    """
    Denoising Diffusion Probabilistic Model (DDPM) implemented as an AutoEncoder.

    This model integrates timestep embeddings into the bottleneck of the AutoEncoder architecture,
    allowing the model to condition on the diffusion timestep during training.

    Parameters
    ----------
    block : Type[Union[UNetBlock, nn.Module]]
        The block type to use in the AutoEncoder (e.g., UNetBlock).
    ch_in : int
        Number of input channels.
    ch_out : int
        Number of output channels.
    width : int
        Base width of the network.
    layers : Sequence[int]
        Number of layers in each block.
    groups : int, optional
        Number of groups for group normalization, by default 1.
    dilation : int, optional
        Dilation rate for convolutions, by default 1.
    norm_layer : Callable[..., nn.Module], optional
        Normalization layer to use, by default nn.BatchNorm2d.
    skip_connect : bool, optional
        Whether to use skip connections, by default False.
    timestep_embed_dim : int, optional
        Dimension of the timestep embedding, by default 256.

    Attributes
    ----------
    timestep_embedding : TimestepEmbedding
        Module for generating timestep embeddings.
    t_embed_proj : nn.Linear
        Linear layer to project timestep embeddings to match the bottleneck dimensions.
    """

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
        timestep_embed_dim: int = 256,  # Dimension of the timestep embedding
    ):
        super(DDPM, self).__init__(block, ch_in, ch_out, width, layers, groups, dilation, norm_layer, skip_connect)

        # Timestep embedding module
        self.timestep_embedding = TimestepEmbedding(timestep_embed_dim)
        # Additional layer to project timestep embeddings to match bottleneck dimensions
        self.t_embed_proj = nn.Linear(timestep_embed_dim, width * 16)
        print(f"DDPM initialized with timestep_embed_dim={timestep_embed_dim}")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for DDPM.

        Parameters
        ----------
        x : torch.Tensor
            Noised image (x_t) of shape (batch_size, channels, height, width).
        t : torch.Tensor
            Timestep tensor of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor of shape (batch_size, channels, height, width).
        """
        # Ensure t is (batch_size,)
        if t.dim() > 1:
            t = t.view(-1)

        self.feature_vectors = []

        # Forward through the encoder and collect feature vectors via hooks
        _ = self.encoder(x)
        if not self.feature_vectors:
            raise ValueError("No feature vectors collected from encoder.")
        feat = self.bottleneck(self.feature_vectors.pop())

        # Timestep embedding
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
            A dictionary containing the loss and the model's output.
        """
        x_t = batch["x_t"]
        t = batch["t"]
        noise = batch.get("noise", None)

        output = self.model(x_t, t)  # Predicted noise
        loss = nn.MSELoss()(output, noise) if noise is not None else None

        return {"loss": loss, "output": output}

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
            A dictionary containing the loss and the model's output.
        """
        return self.train_step(batch=batch)

    def noise_scheduling(self, noise_scheduler: NoiseScheduler) -> None:
        """
        Update the noise scheduler used by the trainer.

        Parameters
        ----------
        noise_scheduler : NoiseScheduler
            New noise scheduler to be used.
        """
        self.noise_scheduler = noise_scheduler

    def generate(
        self, n_samples: int, channels: int, height: int, width: int, intermediate: int = 0, ddim: bool = False
    ) -> Tuple[Tensor, List[Any]]:
        """
        Generate new samples using the trained diffusion model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        channels : int
            Number of channels in the generated images.
        height : int
            Height of the generated images.
        width : int
            Width of the generated images.
        intermediate : int, optional
            Number of intermediate samples to save during generation, by default 0.
        ddim : bool, optional
            Whether to use DDIM sampling. If True, uses DDIM; otherwise, uses DDPM.

        Returns
        -------
        Tuple[Tensor, List[Any]]
            Generated samples tensor of shape (n_samples, channels, height, width).
            List of intermediate samples if specified.
        """
        timesteps = self.noise_scheduler.timesteps
        alpha = self.noise_scheduler.alpha
        alpha_bar = self.noise_scheduler.alpha_bar
        posterior_variance = self.noise_scheduler.posterior_variance

        # Initialize with standard normal noise
        x = torch.randn(n_samples, channels, height, width)

        self.toggle("test")
        save_step = timesteps // intermediate if intermediate > 0 else 0
        save_x = []
        for t_step in reversed(range(timesteps)):
            t_tensor = torch.full((n_samples,), t_step, dtype=torch.long)
            batch = {"x_t": x, "t": t_tensor}
            self.run_batch(batch)
            predict = self.log_test["output"][0]

            if ddim:
                # Deterministic update for DDIM
                if t_step > 0:
                    sqrt_alpha = torch.sqrt(alpha[t_step])
                    sqrt_alpha_prev = torch.sqrt(alpha[t_step - 1])

                    x_prev = (x - self.noise_scheduler.sqrt_one_minus_alpha_bar[t_step] * predict) / sqrt_alpha
                    x_prev = (
                        sqrt_alpha_prev * x_prev + self.noise_scheduler.sqrt_one_minus_alpha_bar[t_step - 1] * predict
                    )
                else:
                    # Final timestep (t=0) handling
                    x_prev = (x - self.noise_scheduler.sqrt_one_minus_alpha_bar[t_step] * predict) / torch.sqrt(
                        alpha[t_step]
                    )
                x = x_prev
            else:
                # DDPM sampling
                t_noise = self.noise_scheduler.beta[t_step]
                if t_step > 0:
                    sqrt_recip_alpha = 1 / torch.sqrt(alpha[t_step])
                    mu_theta = sqrt_recip_alpha * (
                        x - (t_noise / self.noise_scheduler.sqrt_one_minus_alpha_bar[t_step]) * predict
                    )

                    # Sample from the posterior
                    noise = torch.randn_like(x)
                    var = torch.sqrt(posterior_variance[t_step - 1]).view(-1, 1, 1, 1)
                    x = mu_theta + var * noise
                else:
                    x = (x - (t_noise / self.noise_scheduler.sqrt_one_minus_alpha_bar[t_step]) * predict) / torch.sqrt(
                        alpha[t_step]
                    )
            if intermediate > 0 and t_step % save_step == save_step - 1:
                save_x.append(x.clone())

        return x, save_x

    def visualize_samples(
        self,
        final_samples: torch.Tensor,
        intermediate_images: List[Any] = None,
        train_losses: List[float] = None,
        valid_losses: List[float] = None,
        lr_history: List[float] = None,
    ) -> None:
        """
        Visualize generated samples and training progress.

        Parameters
        ----------
        final_samples : torch.Tensor
            Final generated samples. Shape: (n_samples, channels, height, width).
        intermediate_images : List[Any], optional
            Intermediate images for selected samples. Shape: (num_selected, channels, height, width), by default None.
        train_losses : List[float], optional
            Training loss history, by default None.
        valid_losses : List[float], optional
            Validation loss history, by default None.
        lr_history : List[float], optional
            Learning rate history, by default None.
        """
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.2)

        # Upper Left (0:2, 0:2) - 2x2 grid region: 16 generated images with 4x4 grids
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        grid_img = make_grid(final_samples, nrow=4, padding=2, normalize=True)
        grid_img_np = grid_img.permute(1, 2, 0).cpu().numpy()
        ax1.imshow(grid_img_np)
        ax1.set_title("Generated Images (4x4 Grid)")
        ax1.axis("off")

        # Upper Right (0:2, 2:4) - 2x2 grid region: visualization of selected images using intermediate output
        if intermediate_images is not None and len(intermediate_images) > 0:
            # merge intermediate outputs to single grid
            intermediate_grids = []
            for img in intermediate_images:
                steps = torch.stack(img, dim=0)  # (steps, C, H, W)
                steps_grid = make_grid(steps, nrow=1, padding=2, normalize=True)
                intermediate_grids.append(steps_grid)

            # merge multiple grids to single image
            intermediate_grids = torch.stack(intermediate_grids, dim=0)  # (num_samples, C, H, W)
            intermediate_grids = make_grid(intermediate_grids, nrow=len(intermediate_grids), padding=1, normalize=True)
            intermediate_grids_np = intermediate_grids.permute(1, 2, 0).cpu().numpy()

            ax2 = fig.add_subplot(gs[0:2, 2:4])
            ax2.imshow(intermediate_grids_np)
            ax2.set_title("Intermediate Steps of Selected Images")
            ax2.axis("off")

        # Lower (2,0:4) : Line chart of train, valid loss & learning rate history
        ax3 = fig.add_subplot(gs[2, :])
        if train_losses is not None:
            ax3.plot(train_losses, label="Train Loss", color="blue")
        if valid_losses is not None:
            ax3.plot(valid_losses, label="Valid Loss", color="orange")
        if lr_history is not None:
            ax4 = ax3.twinx()  # 두 번째 y축 생성
            ax4.plot(lr_history, label="Learning Rate", color="green")
            ax4.set_ylabel("Learning Rate", color="green")
            ax4.tick_params(axis="y", labelcolor="green")
            ax4.legend(loc="upper right")
        ax3.set_title("Training and Validation Loss with Learning Rate")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.legend(loc="upper left")
        ax3.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # config
    DDIM = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    epoch = 100
    learning_rate = 5e-3
    target_width = target_height = 64
    data_path = "./mnist_example"

    # NoiseScheduler & ForwardDiffusion
    noise_scheduler = NoiseScheduler()
    forward_diffusion = ForwardDiffusion(noise_scheduler=noise_scheduler)
    print("ForwardDiffusion initialized.")

    transform = transforms.Compose(
        [
            transforms.Resize((target_height, target_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    print("Transform defined.")

    dataset = CustomMNISTDataset(root=data_path, train=True, transform=transform, download=True)
    print("CustomMNISTDataset initialized.")

    collator = CollateBase(x_key=["pixel_values"], y_key=[], aux_key=[])
    collator.add_fn(name="forward_diffusion", fn=forward_diffusion)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator.wrap())
    print("DataLoader initialized.")

    model = DDPM(
        block=UNetBlock, ch_in=1, ch_out=1, width=8, layers=[1, 1, 1, 1], skip_connect=True, timestep_embed_dim=256
    )
    print("Model defined.")

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    warm_up_steps = int(0.1 * len(train_loader) * epoch)
    scheduler = SchedulerChain(
        WarmUpScheduler(optimizer=optimizer, total_iters=warm_up_steps),
        CosineAnnealingScheduler(optimizer=optimizer, total_iters=len(train_loader) * epoch - warm_up_steps),
    )
    print("Optimizer and scheduler initialized.")

    # Trainer
    trainer = DDPMTrainer(
        model=model,
        device=device,
        mode="train",
        optimizer=optimizer,
        scheduler=scheduler,
        log_bar=True,
        scheduler_step_on_batch=True,
        gradient_clip=0.0,
    )
    trainer.noise_scheduling(noise_scheduler)
    print("Trainer initialized.")

    # 학습 수행
    print("Starting training...")
    trainer.run(n_epoch=epoch, loader=train_loader, valid_loader=None)
    trainer.save_model(os.path.join("../../ddpm_model.pth"))
    print("Training completed.")
    train_loss = trainer.log_train.get("loss", None)
    valid_loss = trainer.log_test.get("loss", None)

    # 샘플 생성 및 시각화
    print("Generating samples...")
    trainer.toggle("test")
    generated = trainer.generate(
        n_samples=16,
        channels=1,
        height=target_height,
        width=target_width,
        intermediate=4,
        ddim=DDIM,
    )
    generated_samples, generated_intermediate_samples = generated
    # handling intermediate outputs
    generated_intermediate_samples = torch.stack([inter[:4] for inter in generated_intermediate_samples])
    trainer.visualize_samples(
        final_samples=generated_samples,
        intermediate_images=generated_intermediate_samples.permute(1, 0, 2, 3, 4),
        train_losses=trainer.get_loss_history('train'),
        valid_losses=None,
    )
    print("Sample generation and visualization completed.")
    shutil.rmtree(data_path)
