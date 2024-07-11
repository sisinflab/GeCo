import torch 
from torch import nn 
import numpy as np 


class Generator(nn.Module):
    def __init__(
            self,
            conv_filters: list = [64, 128, 256, 512, 512, 512],
        ) -> None:

        super(Generator, self).__init__()
        
        enc_layers = []
        dec_layers = []

        enc_layers.append(nn.Conv2d(3, conv_filters[0], kernel_size=4, stride=2, padding=1))

        in_channels = conv_filters[0]
        for filter in conv_filters[1:-1]:
            enc_layers.append(self._enc_block(in_channels, filter))
            in_channels = filter

        self.final_enc = self._enc_block(in_channels, conv_filters[-1])

        conv_filters.reverse()

        self.m = self._dec_block(in_channels, conv_filters[0])

        for filter in conv_filters[2:]:
            dec_layers.append(self._dec_block(in_channels, filter))
            in_channels = filter

        dec_layers.append(
            self._out_block(in_channels)
        )

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)
        
        del enc_layers, dec_layers

    def _enc_block(
            self,
            in_channels: int,
            out_channels: int
        ) -> nn.Sequential:

        return nn.Sequential(
            nn.LeakyReLU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )
    
    def _dec_block(
            self,
            in_channels: int,
            out_channels: int
        ) -> nn.Sequential:
        
        return nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Dropout(0.5),
            nn.BatchNorm2d(out_channels)
        )
    
    def _out_block(
            self,
            in_channels: int,
        ) -> nn.Sequential:
        
        return nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(
            self,
            x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        enc = self.encoder(x)
        final_enc = self.final_enc(enc)
        d1 = self.m(final_enc)
        dec = self.decoder(d1)
        return enc, d1, dec


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    model = Generator().to(device)
    img = torch.randn(32, 3, 64, 64).to(device)
    enc, d1, dec = model(img)
    print('SUCCESS')
