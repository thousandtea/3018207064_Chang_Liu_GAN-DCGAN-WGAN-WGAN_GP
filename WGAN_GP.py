import torch
import torchvision
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import csv

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
lr = 1e-4
batch_size = 64
image_size = 64
channels_img = 1
z_dim = 100
num_epochs = 50
features_d = 16
features_g = 16
d_iter = 5
lambda_gp = 10

transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]),
    ]
)

mnist = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
csv_file = open('loss/wgangp_loss.csv', 'w', newline='', encoding='gbk')
writer = csv.writer(csv_file)


def gradient_penalty(critic, real, fake, device="cpu"):
    batch_size, C, H, W = real.shape
    alpha = torch.rand((batch_size, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            # no batch norm because of gradient penalty
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            # no batch norm because of gradient penalty
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


def initialize_weights(model):
    for m in model.modules():
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


G = Generator(z_dim, channels_img, features_g).to(device)
D = Discriminator(channels_img, features_d).to(device)
initialize_weights(G)
initialize_weights(D)

g_opt = opt.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
d_opt = opt.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

fixed_noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
step = 0


for epoch in range(num_epochs):
    for i, (real, _) in enumerate(loader):
        # train discriminator
        real = real.to(device)
        cur_batch_size = real.shape[0]

        for _ in range(d_iter):
            disc_real = D(real).view(-1)

            noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
            fake = G(noise)
            disc_fake = D(fake).reshape(-1)

            gp = gradient_penalty(D, real, fake, device=device)

            d_loss = (
                -(torch.mean(disc_real) - torch.mean(disc_fake)) + lambda_gp * gp
            )
            D.zero_grad()
            d_loss.backward(retain_graph=True)
            d_opt.step()

        # train generator
        output = D(fake).view(-1)
        g_loss = -torch.mean(output)
        G.zero_grad()
        g_loss.backward()
        g_opt.step()

        if i % 100 == 0 and i > 0:
            print('Epoch[{}/{}], Batch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                  'D real: {:.6f},D fake: {:.6f}'.format(
                epoch, num_epochs, i, len(loader), d_loss, g_loss,
                disc_real.data.mean(), disc_fake.data.mean()
            ))
            writer.writerow([step, d_loss.item(), g_loss.item()])

            with torch.no_grad():
                fake = G(noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                save_image(img_grid_real, 'img/wgan_gp/wgangp_real_images-{}.png'.format(step))
                save_image(img_grid_fake, 'img/wgan_gp/wgangp_fake_images-{}.png'.format(step))

                step += 1


csv_file.close()
torch.save(G.state_dict(), 'pth/wgangp_generator.pth')
torch.save(D.state_dict(), 'pth/wgangp_discriminator.pth')