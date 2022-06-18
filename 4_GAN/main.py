import sys
import turtle

print(sys.version)
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

print(torch.__version__)  # 1.0.1

import matplotlib.pyplot as plt


def show_imgs(x, new_fig=True):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0, 2).transpose(0, 1)  # channels as last dimension
    if new_fig:
        plt.figure()
    plt.imshow(grid.numpy())
    plt.show()


class Discriminator(torch.nn.Module):
    def __init__(self, inp_dim=784):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), 784)
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.sigmoid(out)
        return out


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 784)

    def forward(self, x):
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.tanh(out)
        out = out.view(out.size(0), 1, 28, 28)
        return out


D = Discriminator()
print(D)
G = Generator()
print(G)

dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST/',
                                            transform=transforms.Compose([transforms.ToTensor(),
                                                                          transforms.Normalize((0.5,), (0.5,))]),
                                            download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

ix = 149
x, _ = dataset[ix]
plt.matshow(x.squeeze().numpy(), cmap=plt.cm.gray)
plt.colorbar()
plt.show()

criterion = nn.BCELoss()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device: ', device)
D = Discriminator().to(device)
G = Generator().to(device)
optimizerD = torch.optim.SGD(D.parameters(), lr=0.03)
optimizerG = torch.optim.SGD(G.parameters(), lr=0.03)
lab_real = torch.ones(64, 1, device=device)
lab_fake = torch.zeros(64, 1, device=device)

# for logging:
collect_x_gen = []
fixed_noise = torch.randn(64, 100, device=device)
fig = plt.figure()  # keep updating this one
plt.ion()


all_losses = []

for epoch in range(3):  # 3 epochs
    size = 0
    current_loss = 0
    for i, data in enumerate(dataloader, 0):
        # STEP 1: Discriminator optimization step
        x_real, _ = iter(dataloader).next()
        x_real = x_real.to(device)
        # reset accumulated gradients from previous iteration
        optimizerD.zero_grad()

        D_x = D(x_real)
        lossD_real = criterion(D_x, lab_real)

        z = torch.randn(64, 100, device=device)  # random noise, 64 samples, z_dim=100
        x_gen = G(z).detach()
        D_G_z = D(x_gen)
        lossD_fake = criterion(D_G_z, lab_fake)
        lossD = lossD_real + lossD_fake

        current_loss += lossD.cpu().detach().numpy()
        size += len(lab_real) + len(lab_fake)

        lossD.backward()
        optimizerD.step()

        # STEP 2: Generator optimization step
        # reset accumulated gradients from previous iteration
        optimizerG.zero_grad()

        z = torch.randn(64, 100, device=device)  # random noise, 64 samples, z_dim=100
        x_gen = G(z)
        D_G_z = D(x_gen)
        lossG = criterion(D_G_z, lab_real)  # -log D(G(z))

        lossG.backward()
        optimizerG.step()
        if i % 100 == 0:
            x_gen = G(fixed_noise)
            # show_imgs(x_gen, new_fig=False)
            fig.canvas.draw()
            print('e{}.i{}/{} last mb D(x)={:.4f} D(G(z))={:.4f}'.format(
                epoch, i, len(dataloader), D_x.mean().item(), D_G_z.mean().item()))

    all_losses.append(current_loss/size)
    # End of epoch
    x_gen = G(fixed_noise)
    collect_x_gen.append(x_gen.detach().clone())

for x_gen in collect_x_gen:
    show_imgs(x_gen)

plt.figure()
plt.plot(all_losses)
plt.show()

random_noise = torch.randn(8, 100, device=device)
x_gen = G(random_noise)
show_imgs(x_gen, new_fig=False)

random_list = [0.1,1,-5]
for cnt in range(3):
    for random_num in range(5):
        for _ in random_noise:
            _[random_num] = random_list[cnt]
        x_gen = G(random_noise)
        show_imgs(x_gen, new_fig=False)