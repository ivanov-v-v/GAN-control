import torch

CONV_INIT_NORMAL_PARAMS = (0, 0.02)
BATCHNORM_INIT_NORMAL_PARAMS = (1, 0.02)
BATCHNORM_CONSTANT_INIT_VAL = 0


def initialize_weights(layer: torch.nn.Module):
    """ Initialization suggested in the original DCGAN paper """
    layer_name = layer.__class__.__name__
    if 'Conv' in layer_name:
        layer: torch.nn.Conv2d = layer
        torch.nn.init.normal_(layer.weight.data, *CONV_INIT_NORMAL_PARAMS)
    elif 'BatchNorm' in layer_name:
        layer: torch.nn.BatchNorm2d = layer
        torch.nn.init.normal_(layer.weight.data, *BATCHNORM_INIT_NORMAL_PARAMS)
        torch.nn.init.constant_(layer.bias.data, BATCHNORM_CONSTANT_INIT_VAL)


class Generator(torch.nn.Module):
    def __init__(self, latent_space_dim: int, base_features_depth: int):
        super(Generator, self).__init__()
        self._layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=latent_space_dim,
                out_channels=base_features_depth * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            torch.nn.BatchNorm2d(base_features_depth * 8),
            torch.nn.ReLU(inplace=True),
            # B x 512 x 4 x 4
            torch.nn.ConvTranspose2d(
                in_channels=base_features_depth * 8,
                out_channels=base_features_depth * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(base_features_depth * 4),
            torch.nn.ReLU(inplace=True),
            # B x 256 x 8 x 8
            torch.nn.ConvTranspose2d(
                in_channels=base_features_depth * 4,
                out_channels=base_features_depth * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(base_features_depth * 2),
            torch.nn.ReLU(inplace=True),
            # B x 128 x 16 x 16
            torch.nn.ConvTranspose2d(
                in_channels=base_features_depth * 2,
                out_channels=base_features_depth,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(base_features_depth),
            torch.nn.ReLU(inplace=True),
            # B x 64 x 32 x32
            torch.nn.ConvTranspose2d(
                in_channels=base_features_depth,
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
        return self._layers(latent_vector)


class Discriminator(torch.nn.Module):
    def __init__(self, base_features_depth: int):
        super(Discriminator, self).__init__()
        self._layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=base_features_depth,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(base_features_depth),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # B x 64 x 32 x 32
            torch.nn.Conv2d(
                in_channels=base_features_depth,
                out_channels=base_features_depth * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(base_features_depth * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # B x 128 x 16 x 16
            torch.nn.Conv2d(
                in_channels=base_features_depth * 2,
                out_channels=base_features_depth * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(base_features_depth * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # B x 256 x 8 x 8
            torch.nn.Conv2d(
                in_channels=base_features_depth * 4,
                out_channels=base_features_depth * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(base_features_depth * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # B x 512 x 4 x 4
            torch.nn.Conv2d(
                in_channels=base_features_depth * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            torch.nn.Sigmoid(),
            # B x 1 x 1 x 1
        )

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        assert input_image.ndim == 4 and input_image.size()[1] == 3, (
            'Expected shape: (B, 3, H, W)\n'
            f'Received shape: {input_image.size()}'
        )
        return self._layers(input_image)
