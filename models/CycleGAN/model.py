import torch


def initialize_weights(layer: torch.nn.Module):
    """ Initialization suggested in the original CycleGAN paper """
    raise NotImplementedError()


class Generator(torch.nn.Module):
    def __init__(self, conv_dim: int = 64):
        super(Generator, self).__init__()


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
