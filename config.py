from dataclasses import dataclass

@dataclass
class DCGANConfig:
    dataroot = "/notebooks/data"
    workers = 6
    image_size = 64
    batch_size = 64

    num_epochs = 500
    learning_rate = 0.0002
    beta1 = 0.5

    latent_dim = 256
    hidden_generator = 128
    hidden_discriminator = 128

DCGAN_config = DCGANConfig()