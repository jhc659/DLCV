import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output",
    "-o",
    type=str,
    default="./WGAN/output",
    help="path to the directory for output image",
)
parser.add_argument(
    "--ckpt_path",
    "-ckpt",
    type=str,
    default="./GAN.pth",
    help="path to load checkpoint",
)

args = parser.parse_args()


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DCGANDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        # 1. Load the image
        img = torchvision.io.read_image(fname)
        # 2. Resize and normalize the images using torchvision.
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def add_sn(model):
    for name, layer in model.named_children():
        model.add_module(name, add_sn(layer))
        if isinstance(model, (nn.Conv2d, nn.Linear)):
            return nn.utils.spectral_norm(model)
        else:
            return model
    return model


class Generator(nn.Module):
    """
    Input shape: (N, in_dim)
    Output shape: (N, 3, 64, 64)
    """

    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(True),
            )

        self.l1 = nn.Sequential(
            # nn.Linear(in_dim, dim*8*4*4, bias=False),
            nn.ConvTranspose2d(in_dim, dim * 16, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(dim * 16),
            nn.ReLU(True),
        )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 16, dim * 8),
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            nn.ConvTranspose2d(
                dim * 2,
                3,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        # y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):
    """
    Input shape: (N, 3, 64, 64)
    Output shape: (N, )
    """

    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim, h, w):
            return nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv2d(
                        in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=False
                    )
                ),
                nn.LayerNorm([out_dim, h, w]),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.ls = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_dim, dim, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2, 16, 16),
            conv_bn_lrelu(dim * 2, dim * 4, 8, 8),
            conv_bn_lrelu(dim * 4, dim * 8, 4, 4),
            nn.utils.spectral_norm(nn.Conv2d(dim * 8, 1, kernel_size=4, bias=False)),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y


def main():
    z_dim = 100
    G = Generator(z_dim)
    G.load_state_dict(torch.load(args.ckpt_path))
    G.eval()
    G.cuda()
    # Generate 1000 images and make a grid to save them.
    same_seeds(30)
    n_output = 1000
    z_sample = Variable(torch.randn(n_output, z_dim, 1, 1)).cuda()
    imgs_sample = (G(z_sample).data + 1) / 2.0

    # Save the generated images.
    outputPath = args.output
    print(outputPath)
    os.makedirs(outputPath, exist_ok=True)
    for i in range(1000):
        torchvision.utils.save_image(
            imgs_sample[i], os.path.join(outputPath, "{}.png".format(i + 1))
        )


if __name__ == "__main__":
    main()
