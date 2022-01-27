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


lr = 3e-4  # may best
z_dim = 64  # latent noise 128,256 is ok
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 50
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ]
)
mnist = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)
csv_file = open('loss/gan_loss.csv', 'w', newline='', encoding='gbk')
writer = csv.writer(csv_file)


class Generator(nn.Module):
    def __init__(self, z_dim, image_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, image_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.dis = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),   #output should be 0 or 1
        )

    def forward(self, x):
        return self.dis(x)


D = Discriminator(image_dim).to(device)
G = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn(batch_size, z_dim).to(device)


d_opt = opt.Adam(D.parameters(), lr=lr)
g_opt = opt.Adam(G.parameters(), lr=lr)
criterion = nn.BCELoss() #single point crossentropy


step = 0

for epoch in range(num_epochs):
    for i, (real, _) in enumerate(loader):
        # train discriminator
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # loss of real
        disc_real = D(real).view(-1)
        d_real_loss = criterion(disc_real, torch.ones_like(disc_real)) # 1 label

        # loss of fake
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = G(noise)
        disc_fake = D(fake).view(-1)
        d_fake_loss = criterion(disc_fake, torch.zeros_like(disc_fake))  # 0 label

        # loss function and optimize
        d_loss = (d_real_loss + d_fake_loss) / 2
        D.zero_grad()
        d_loss.backward(retain_graph=True)
        d_opt.step()

        # train generator
        output = D(fake).view(-1)
        g_loss = criterion(output, torch.ones_like(output))  # 1 label
        
        #bp and optimize
        G.zero_grad()
        g_loss.backward()
        g_opt.step()

        if i == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                  'D real: {:.6f},D fake: {:.6f}'.format(
                epoch, num_epochs, d_loss, g_loss,
                disc_real.data.mean(), disc_fake.data.mean()
            ))
            writer.writerow([step, d_loss.item(), g_loss.item()])

            with torch.no_grad():
                fake = G(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                save_image(img_grid_real, 'img/gan/gan_real_images-{}.png'.format(step))
                save_image(img_grid_fake, 'img/gan/gan_fake_images-{}.png'.format(step))

                step += 1


csv_file.close()
torch.save(G.state_dict(), 'pth/gan_generator.pth')
torch.save(D.state_dict(), 'pth/gan_discriminator.pth')