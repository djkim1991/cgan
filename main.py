import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable

from model.Discriminator import Discriminator
from model.Generator import Generator

from loaders.MNISTLoader import MNIST
from util.ImageUtil import ImageUtil

# create model objects
discriminator = Discriminator()
generator = Generator()

# set data loader
dataLoader = MNIST()
train_loader, test_loader = dataLoader.train_loader, dataLoader.test_loader

D_optimizer = Adam(params=discriminator.parameters(), lr=0.001)
G_optimizer = Adam(params=generator.parameters(), lr=0.001)
D_loss_function = nn.BCELoss()
G_loss_function = nn.BCELoss()

imageUtil = ImageUtil()

epoch_size = 10000
for epoch in range(epoch_size):
    for i, data in enumerate(train_loader):
        real_data, real_label = data
        real_data, real_label = Variable(real_data), Variable(real_label)

        if torch.cuda.is_available():
            real_data = real_data.cuda()
            real_label = real_label.cuda()

        # target_real = [1, 1, .., 1], target_fake = [0, 0, .., 0]
        batch_size = real_data.size(0)
        target_real = torch.ones(batch_size, dtype=torch.float)
        target_fake = torch.zeros(batch_size, dtype=torch.float)

        if torch.cuda.is_available():
            target_real = target_real.cuda()
            target_fake = target_fake.cuda()

        # for make Fake data
        noise_size = 100
        z = torch.randn(batch_size, noise_size)
        fake_label = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size)))

        if torch.cuda.is_available():
            z = z.cuda()
            fake_label = fake_label.cuda()
            discriminator = discriminator.cuda()
            generator = generator.cuda()

        if i % 1 == 0:
            # Training D with Real Data
            D_optimizer.zero_grad()
            D_real_decision = discriminator.forward(real_data, real_label)
            D_real_loss = D_loss_function(D_real_decision, target_real)

            # Training D with Fake Data
            fake_data = generator.forward(z, fake_label)
            D_fake_decision = discriminator.forward(fake_data, fake_label)
            D_fake_loss = D_loss_function(D_fake_decision, target_fake)

            D_loss = D_real_loss + D_fake_loss
            if i % 10 == 0:
                print('{0}: D_loss is {1}'.format(i, D_loss))
            D_loss.backward()

            D_optimizer.step()

        if i % 1 == 0:
            # Training G based on D's decision
            G_optimizer.zero_grad()

            z = torch.randn(batch_size, noise_size)
            # fake_label = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size)))
            fake_label = Variable(torch.LongTensor(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]*10)))

            if torch.cuda.is_available():
                z = z.cuda()
                fake_label = fake_label.cuda()

            fake_data = generator.forward(z, fake_label)
            D_fake_decision = discriminator.forward(fake_data, fake_label)

            G_loss = G_loss_function(D_fake_decision, target_real)

            if i % 10 == 0:
                print('{0}: G_loss is {1}'.format(i, G_loss))
            if i % 1000 == 0:
                fake_data = fake_data.view(100, -1)
                imageUtil.save(fake_data)

            G_loss.backward()

            G_optimizer.step()
