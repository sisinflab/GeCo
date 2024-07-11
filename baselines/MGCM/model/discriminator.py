import torch 
from torch import nn 
import numpy as np 

class Discriminator(nn.Module):
    def __init__(
            self,
            conv_filters: list = [64, 128, 256, 512]
        ) -> None:

        super(Discriminator, self).__init__()

        disc_layers = []

        in_channels = 3

        disc_layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels * 2, conv_filters[0], kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU()
                )
        )

        in_channels = conv_filters[0]
        for filter in conv_filters[1:]:
            disc_layers.append(self._discriminator_block(in_channels, filter))
            in_channels = filter

        self.encoder = nn.Sequential(*disc_layers)
        self.out_layer = nn.Linear(conv_filters[-1] * 4 * 4, 1)
        del disc_layers
    
    def _discriminator_block(
            self,
            in_channels: int, 
            out_channels: int
        ) -> nn.Sequential:
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(
            self,
            image: torch.Tensor,
            label: torch.Tensor, 
        ) -> torch.Tensor:
        
        x = torch.cat([image, label], dim=1)
        enc = self.encoder(x)
        b, _, _, _ = enc.shape

        enc = enc.view(b, -1)
        lin = self.out_layer(enc)

        return torch.sigmoid(lin), lin
    

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Discriminator().to(device)
    img = torch.randn(16, 3, 64, 64).to(device)
    act, lin = model(img, img)
    print('SUCCESS')
