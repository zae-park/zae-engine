import unittest
from typing import Type, Union, List, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from zae_engine.models import AutoEncoder, VAE
from zae_engine.nn_night.blocks import UNetBlock


class TestAutoEncoder(unittest.TestCase):
    def setUp(self):
        # Define hyperparameters
        self.ch_in = 3
        self.ch_out = 3
        self.width = 64
        self.layers = [2, 2, 2, 2]  # Layer configuration for each stage
        self.image_size = (32, 3, 64, 64)  # (batch_size, channels, height, width)

        # Define model without skip connections
        self.model_no_skip = AutoEncoder(
            block=UNetBlock,
            ch_in=self.ch_in,
            ch_out=self.ch_out,
            width=self.width,
            layers=self.layers,
            skip_connect=False,
        )

        # Define model with skip connections (U-Net style)
        self.model_with_skip = AutoEncoder(
            block=UNetBlock,
            ch_in=self.ch_in,
            ch_out=self.ch_out,
            width=self.width,
            layers=self.layers,
            skip_connect=True,
        )

        # Create sample input
        self.input_tensor = torch.randn(self.image_size)

    def test_forward_no_skip(self):
        # Test forward pass without skip connections
        output = self.model_no_skip(self.input_tensor)
        self.assertEqual(
            output.shape, self.image_size, "Output shape should match input shape without skip connections"
        )

    def test_forward_with_skip(self):
        # Test forward pass with skip connections
        output = self.model_with_skip(self.input_tensor)
        self.assertEqual(output.shape, self.image_size, "Output shape should match input shape with skip connections")

    def test_decoder_input_channels_with_skip_connection(self):
        # AutoEncoder with skip connections
        model_skip = AutoEncoder(
            block=UNetBlock,
            ch_in=self.ch_in,
            ch_out=self.ch_out,
            width=self.width,
            layers=self.layers,
            skip_connect=True,
        )

        # Forward pass to register hooks and pop feature maps
        input_data = torch.randn(1, self.ch_in, 64, 64)
        model_skip(input_data)

        # Ensure that when skip connections are used, the input channels for the decoder layers are correctly increased
        for up_pool, dec in zip(model_skip.up_pools, model_skip.decoder):
            # Skip connections should concatenate feature maps, doubling the input channels for each decoder stage
            input_channels = dec[0].conv1.in_channels
            self.assertEqual(
                input_channels,
                up_pool.out_channels * 2,
                f"Decoder input channels should be doubled due to skip connections at layer {dec}",
            )


class TestVAE(unittest.TestCase):
    """Unit tests for the VAE class in zae_engine.preprocessing."""
    
    def setUp(self):
        """Set up common test data and VAE instance."""
        # VAE 파라미터 설정
        self.block = UNetBlock  # 또는 다른 블록 타입
        self.ch_in = 3
        self.ch_out = 3
        self.width = 64
        self.layers = [2, 2, 2, 2]
        self.groups = 1
        self.dilation = 1
        self.norm_layer = nn.BatchNorm2d
        self.skip_connect = True
        self.latent_dim = 128
        
        # VAE 인스턴스 생성
        self.vae = VAE(
            block=self.block,
            ch_in=self.ch_in,
            ch_out=self.ch_out,
            width=self.width,
            layers=self.layers,
            groups=self.groups,
            dilation=self.dilation,
            norm_layer=self.norm_layer,
            skip_connect=self.skip_connect,
            latent_dim=self.latent_dim
        )
        
        # 테스트 데이터 생성 (배치 크기 4, 채널 3, 64x64 이미지)
        self.batch_size = 4
        self.channels = self.ch_in
        self.height = 64
        self.width_img = 64
        self.test_input = torch.randn(self.batch_size, self.channels, self.height, self.width_img)
    
    def test_forward_pass(self):
        """Test that the VAE forward pass returns reconstructed, mu, and logvar."""
        reconstructed, mu, logvar = self.vae(self.test_input)
        
        # 출력이 모두 반환되는지 확인
        self.assertIsInstance(reconstructed, Tensor)
        self.assertIsInstance(mu, Tensor)
        self.assertIsInstance(logvar, Tensor)
    
    def test_output_shapes(self):
        """Test that the output shapes of reconstructed, mu, and logvar are correct."""
        reconstructed, mu, logvar = self.vae(self.test_input)
        
        # 재구성된 출력의 형태가 입력과 동일한지 확인
        self.assertEqual(reconstructed.shape, self.test_input.shape)
        
        # mu와 logvar의 형태가 (batch_size, latent_dim)인지 확인
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
    
    def test_skip_connections_enabled(self):
        """Test VAE behavior when skip connections are enabled."""
        # VAE 인스턴스를 skip_connect=True으로 초기화
        self.vae.skip_connect = True
        
        # 전방 패스 수행
        reconstructed, mu, logvar = self.vae(self.test_input)
        
        # feature_vectors가 비어있는지 확인
        self.assertEqual(len(self.vae.feature_vectors), 0)
        
        # 추가적인 검증이 필요할 경우 여기에 추가
    
    def test_skip_connections_disabled(self):
        """Test VAE behavior when skip connections are disabled."""
        # VAE 인스턴스를 skip_connect=False으로 설정
        self.vae.skip_connect = False
        
        # 전방 패스 수행
        reconstructed, mu, logvar = self.vae(self.test_input)
        
        # feature_vectors가 비어있는지 확인
        self.assertEqual(len(self.vae.feature_vectors), 0)
        
        # 추가적인 검증이 필요할 경우 여기에 추가
    
    def test_reparameterize(self):
        """Test the reparameterization trick."""
        mu = torch.zeros(self.batch_size, self.latent_dim)
        logvar = torch.zeros(self.batch_size, self.latent_dim)
        
        z = self.vae.reparameterize(mu, logvar)
        
        # z의 형태가 (batch_size, latent_dim)인지 확인
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))
        
        # 평균이 mu와 같고 분산이 1인지 확인 (mu=0, logvar=0일 때)
        self.assertTrue(torch.allclose(z.mean(dim=0), mu.mean(dim=0), atol=1e-5))
        self.assertTrue(torch.allclose(z.var(dim=0, unbiased=False), torch.ones(self.latent_dim), atol=1e-5))
    
    def test_invalid_input_shape(self):
        """Test that VAE raises an error for invalid input shapes."""
        # 잘못된 입력 형태 (예: 3D 텐서)
        invalid_input = torch.randn(self.batch_size, self.channels, self.height)  # Shape: (batch_size, channels, height)
        
        with self.assertRaises(RuntimeError):
            self.vae(invalid_input)
    
    def test_latent_dim(self):
        """Test that changing latent_dim affects mu and logvar dimensions."""
        # latent_dim 변경
        new_latent_dim = 256
        self.vae.latent_dim = new_latent_dim
        self.vae.fc_mu = nn.Linear(self.vae.encoder.encoder.output_dim, new_latent_dim)
        self.vae.fc_logvar = nn.Linear(self.vae.encoder.encoder.output_dim, new_latent_dim)
        
        # 전방 패스 수행
        reconstructed, mu, logvar = self.vae(self.test_input)
        
        # mu와 logvar의 형태가 변경된 latent_dim과 일치하는지 확인
        self.assertEqual(mu.shape, (self.batch_size, new_latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, new_latent_dim))
    
    def test_reconstruction_quality(self):
        """Test that the reconstruction loss decreases after a training step."""
        # 손실 함수 정의
        def vae_loss(reconstructed: Tensor, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
            recon_loss = nn.functional.binary_cross_entropy(reconstructed, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + kl_loss
        
        # 옵티마이저 설정
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        
        # 초기 손실 계산
        reconstructed, mu, logvar = self.vae(self.test_input)
        initial_loss = vae_loss(reconstructed, self.test_input, mu, logvar).item()
        
        # 역전파 및 업데이트
        optimizer.zero_grad()
        loss = vae_loss(reconstructed, self.test_input, mu, logvar)
        loss.backward()
        optimizer.step()
        
        # 업데이트 후 손실 계산
        reconstructed, mu, logvar = self.vae(self.test_input)
        updated_loss = vae_loss(reconstructed, self.test_input, mu, logvar).item()
        
        # 손실이 감소했는지 확인
        self.assertLess(updated_loss, initial_loss)
    
    def test_generated_output(self):
        """Test that generated output from random latent vectors has the correct shape."""
        # 잠재 공간에서 랜덤 샘플링
        z = torch.randn(self.batch_size, self.latent_dim)
        
        # Bottleneck을 통과하여 디코더 입력 생성
        feat = self.vae.bottleneck(z)
        
        # 디코더를 통해 재구성
        for up_pool, dec in zip(self.vae.up_pools, self.vae.decoder):
            feat = up_pool(feat)
            if self.vae.skip_connect and len(self.vae.feature_vectors) > 0:
                feat = torch.cat((feat, self.vae.feature_vectors.pop()), dim=1)
            feat = dec(feat)
        
        # 최종 출력
        generated = self.vae.sig(self.vae.fc(feat))
        
        # 생성된 출력의 형태가 입력과 동일한지 확인
        self.assertEqual(generated.shape, self.test_input.shape)

if __name__ == '__main__':
    unittest.main()