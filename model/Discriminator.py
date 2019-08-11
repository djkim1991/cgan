'''
    writer: dororongju
    github: https://github.com/djkim1991/CGAN/issues/1
'''
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=28*28 + 10, out_features=512),
            nn.LeakyReLU(0.2),
            nn.Dropout()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.LeakyReLU(0.2),
            nn.Dropout()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        """
        :param x:   input tensor    [batch_size * 1 * 28 * 28]
        :return:    possibility of that the image is real data
        """
        x = x.view(-1, 28*28)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
