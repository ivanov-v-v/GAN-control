import torch


CONV_INIT_NORMAL_PARAMS = (0, 0.02)
CONV_BIAS_INIT_VAL = 0


def initialize_weights(layer: torch.nn.Module):
    """ Initialization suggested in the original DCGAN paper """
    layer_name = layer.__class__.__name__
    if 'Conv' in layer_name:
        layer: torch.nn.Conv2d = layer
        torch.nn.init.normal_(layer.weight.data, *CONV_INIT_NORMAL_PARAMS)


class ResNetBlock(torch.nn.Module):
    def __init__(self, n_channels: int):
        super(ResNetBlock, self).__init__()
        self._layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            torch.nn.InstanceNorm2d(n_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            torch.nn.InstanceNorm2d(n_channels),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self._layers(input)


class Generator(torch.nn.Module):
    def __init__(
            self, n_resnet_blocks: int = 9, padding_mode: str = 'reflect',
    ):
        assert n_resnet_blocks > 0

        super(Generator, self).__init__()
        self._layers = torch.nn.Sequential(
            # Input: 3 x 256 x 256
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode=padding_mode,
                bias=True,
            ),
            torch.nn.InstanceNorm2d(64, affine=False),
            torch.nn.ReLU(inplace=True),
            # 64 x 256 x 256
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode=padding_mode,
                bias=True,
            ),
            torch.nn.InstanceNorm2d(128, affine=False),
            torch.nn.ReLU(inplace=True),
            # 128 x 128 x 128
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode=padding_mode,
                bias=True,
            ),
            torch.nn.InstanceNorm2d(256, affine=False),
            torch.nn.ReLU(inplace=True),
            # 256 x 64 x 64
            *[ResNetBlock(n_channels=256) for _ in range(n_resnet_blocks)],
            torch.nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=True,
            ),
            torch.nn.InstanceNorm2d(128, affine=False),
            torch.nn.ReLU(inplace=True),
            # 128 x 128 x 128
            torch.nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=True,
            ),
            torch.nn.InstanceNorm2d(64, affine=False),
            torch.nn.ReLU(inplace=True),
            # 64 x 256 x 256
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=3,
                kernel_size=7,
                stride=1,
                padding=0,
                padding_mode=padding_mode,
                bias=True,
            ),
            torch.nn.InstanceNorm2d(3, affine=False),
            torch.nn.Tanh(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._layers(input)


class Discriminator(torch.nn.Module):
    """
    So-called 'patchGAN' architecture: output is comprised of a 30x30
    value matrix, each representing result of classifying 70x70 patch on the
    input image.  
    """

    def __init__(self, padding_mode: str = 'reflect'):
        super(Discriminator, self).__init__()
        self._layers = torch.nn.Sequential(
            # 3 x 256 x 256
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode=padding_mode,
                bias=True,
            ),
            torch.nn.InstanceNorm2d(64, affine=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # 64 x 128 x 128
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode=padding_mode,
                bias=True,
            ),
            torch.nn.InstanceNorm2d(128, affine=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # 128 x 64 x 64
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode=padding_mode,
                bias=True,
            ),
            torch.nn.InstanceNorm2d(256, affine=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # 256 x 32 x 32
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode=padding_mode,
                bias=True,
            ),
            torch.nn.InstanceNorm2d(512, affine=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # 512 x 31 x 31
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode=padding_mode,
                bias=True,
            ),
            # 1 x 30 x 30
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._layers(input)
