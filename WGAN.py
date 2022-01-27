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
lr = 5e-5
z_dim = 128
image_size = 64
batch_size = 64
num_epochs = 50
features_d = 64
features_g = 64
d_iter = 5
clip_weight = 0.01
channels_img = 1

transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
        ),
    ]
)

mnist = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
csv_file = open('loss/wgan_loss.csv', 'w', newline='', encoding='gbk')
writer = csv.writer(csv_file)

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # compared to DCGAN, remove the Sigmoid at last
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.dis(x)


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

g_opt = opt.RMSprop(G.parameters(), lr=lr)
d_opt = opt.RMSprop(D.parameters(), lr=lr)


fixed_noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
step = 0

for epoch in range(num_epochs):
    for i, (real, _) in enumerate(loader):
        # train discriminator
        real = real.to(device)
        cur_batch_size = real.shape[0]

        for _ in range(d_iter):
            # train discriminator
            disc_real = D(real).view(-1)

            noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
            fake = G(noise)
            disc_fake = D(fake).view(-1)
            # different from DCGAN, no log in loss
            d_loss = -(torch.mean(disc_real) - torch.mean(disc_fake))

            D.zero_grad()
            d_loss.backward(retain_graph=True)
            d_opt.step()

            # to satisfy lipschitz continual condition, clip weights shoule be between -0.01, 0.01
            for p in D.parameters():
                p.data.clamp_(-clip_weight, clip_weight)

        # train generator
        output = D(fake).reshape(-1)
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

                save_image(img_grid_real, 'img/wgan/wgan_real_images-{}.png'.format(step))
                save_image(img_grid_fake, 'img/wgan/wgan_fake_images-{}.png'.format(step))

                step += 1


csv_file.close()
torch.save(G.state_dict(), 'pth/wgan_generator.pth')
torch.save(D.state_dict(), 'pth/wgan_discriminator.pth')