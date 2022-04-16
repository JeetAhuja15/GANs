import torch
import torch.nn as nn

import torch.nn.utils.spectral_norm

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x channels_img x 32x32
            nn.utils.spectral_norm(nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            )),  # 16 x 16
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # 8x8
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 4x4

            nn.utils.spectral_norm(nn.Conv2d(features_d * 4, 1, kernel_size=4, stride=2, padding=0)),  # 1x1
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


# Generator

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 8, 4, 1, 0),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),

            nn.utils.spectral_norm(nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1)),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)

# Initialize the weights
def init_weights(model):
  for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
      nn.init.normal_(m.weight.data, 0.0, 0.02)

#doubt --> most prob cleared
def test():
  N, in_channels, H, W = 8, 3, 32, 32
  z_dim = 100
  x = torch.randn((N, in_channels, H, W))
  disc = Discriminator(in_channels, 8)
  init_weights(disc)
  assert disc(x).shape == (N, 1, 1, 1)
  gen = Generator(z_dim, in_channels, 8)
  init_weights(gen)
  z = torch.randn((N, z_dim, 1, 1))
  assert gen(z).shape == (N, in_channels, H, W)
