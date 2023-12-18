from dataclasses import dataclass

@dataclass
class DCGANConfig:
    dataroot = "/notebooks/data"
    workers = 6
    image_size = 64
    batch_size = 64

    num_epochs = 50
    learning_rate = 0.0002
    beta1 = 0.5

    latent_dim = 100
    hidden_generator = 64
    hidden_discriminator = 64

DCGAN_config = DCGANConfig()