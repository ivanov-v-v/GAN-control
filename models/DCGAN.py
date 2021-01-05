import torch


def initialize_weights(layer: torch.nn.Module):
    layer_name = layer.__class__.__name__
    if 'Conv' in layer_name:
        layer: torch.nn.Conv2d = layer
        torch.nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in layer_name:
        layer: torch.nn.BatchNorm2d = layer
        torch.nn.init.normal_(layer.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(layer.bias.data, 0)


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self._net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=100,
                out_channels=1024,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            # B x 1024 x 4 x 4
            torch.nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            # B x 512 x 8 x 8
            torch.nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            # B x 256 x 16 x 16
            torch.nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            # B x 128 x 32 x32
            torch.nn.ConvTranspose2d(
                in_channels=128,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.Tanh(),
            # B x 3 x 64 x 64
        )

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        assert latent_vector.ndim == 4, (
            'Expected shape: (B, C, 1, 1)\n'
            f'Received shape: {latent_vector.size()}'
        )
        return self._net(latent_vector)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
