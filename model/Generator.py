'''
    writer: dororongju
    github: https://github.com/djkim1991/CGAN/issues/1
'''
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=100 + 10, out_features=512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=28*28),
            # nn.Tanh()
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        """
        :param x:   input tensor    [batch_size * noise_size]
        :return:    output tensor   [batch_size * 1 * 28 * 28]
        """

        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        x = self.layer1(x)
        x = self.layer2(x)

        return x.view(-1, 1, 28, 28)
