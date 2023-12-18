import torchvision.datasets as dset
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import wandb
import os
import warnings

from torch.utils.data import DataLoader, Dataset
from itertools import repeat
from tqdm import tqdm
from piq import ssim, FID

from config import DCGAN_config
from model import Generator, Discriminator, weights_init

warnings.filterwarnings("ignore")

def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader

def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))

def collate_fn(batch):
    return {"images": torch.stack(batch)}

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def main():
    dataset = dset.ImageFolder(root=DCGAN_config.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(DCGAN_config.image_size),
                               transforms.CenterCrop(DCGAN_config.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataloader = DataLoader(dataset, batch_size=DCGAN_config.batch_size,
                            shuffle=True, num_workers=DCGAN_config.workers)
    
    device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
    netG = Generator(DCGAN_config.latent_dim, DCGAN_config.hidden_generator).to(device)
    netG.apply(weights_init)
    netD = Discriminator(DCGAN_config.hidden_discriminator).to(device)
    netD.apply(weights_init)

    criterion = torch.nn.BCELoss()
    fixed_noise = torch.randn(64, DCGAN_config.latent_dim, 1, 1, device=device)

    optimizerD = torch.optim.Adam(netD.parameters(), lr=DCGAN_config.learning_rate, betas=(DCGAN_config.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=DCGAN_config.learning_rate, betas=(DCGAN_config.beta1, 0.999))

    wandb.init(project="DCGAN")

    if not os.path.exists("saved"):
        os.mkdir("saved")

    for epoch in tqdm(range(DCGAN_config.num_epochs)):

        D_x_log = []
        D_G_z1_log = []
        D_G_z2_log = []
        errD_log = []
        errG_log = []

        for i, data in enumerate(dataloader):

            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label.float())
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(b_size, DCGAN_config.latent_dim, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label.float())
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(1)
            output = netD(fake).view(-1)
            errG = criterion(output, label.float())
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            D_x_log.append(D_x)
            D_G_z1_log.append(D_G_z1)
            D_G_z2_log.append(D_G_z2)
            errD_log.append(errD.item())
            errG_log.append(errG.item())


        log_output = {}

        log_output['D_x'] = sum(D_x_log) / len(D_x_log)
        log_output['D_G_z1'] = sum(D_G_z1_log) / len(D_G_z1_log)
        log_output['D_G_z2'] = sum(D_G_z2_log) / len(D_G_z2_log)
        log_output['errD'] = sum(errD_log) / len(errD_log)
        log_output['errG'] = sum(errG_log) / len(errG_log)


        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            image = vutils.make_grid(fake, padding=2, normalize=True)
            log_output['image'] = wandb.Image(image)

            b = next(iter(dataloader))
            real_cpu = b[0].to(device)

            real_c = real_cpu.detach().cpu().clone()
            fake_c = fake.detach().cpu().clone()
            if real_c.shape[0] == fake_c.shape[0]:
                norm_ip(real_c, float(real_c.min()), float(real_c.max()))
                norm_ip(fake_c, float(fake_c.min()), float(fake_c.max()))
                log_output['SSIM'] = ssim(real_c, fake_c).item()

                dataset_real = CustomDataset(real_c)
                dataset_fake = CustomDataset(fake_c)
                dataloader_real = DataLoader(dataset_real, batch_size=64, collate_fn=collate_fn)
                dataloader_fake = DataLoader(dataset_fake, batch_size=64, collate_fn=collate_fn)
                fid_metric = FID()
                first_feats = fid_metric.compute_feats(dataloader_real)
                second_feats = fid_metric.compute_feats(dataloader_fake)
                log_output['FID'] = fid_metric(first_feats, second_feats)

        wandb.log(log_output)
        torch.save(netG.state_dict(), f"saved/generator_{epoch}.pth")
        torch.save(netD.state_dict(), f"saved.discriminator_{epoch}.pth")
    wandb.finish()
                
if __name__ == "__main__":
    main()