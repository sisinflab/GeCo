import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            features: int = 64
        ) -> None:
        
        super(Generator, self).__init__()

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )  # 128 X 128
        
        self.down1 = self._block(features, features*2, down=True, act="leaky", use_dropout=False)
        self.down2 = self._block(features*2, features*4, down=True, act="leaky", use_dropout=False)
        self.down3 = self._block(features*4, features*8, down=True, act="leaky", use_dropout=False)
        self.down4 = self._block(features*8, features*8, down=True, act="leaky", use_dropout=False)
        self.down5 = self._block(features*8, features*8, down=True, act="leaky", use_dropout=False)
        self.down6 = self._block(features*8, features*8, down=True, act="leaky", use_dropout=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8,features*8,4,2,1,padding_mode="reflect"),
            nn.ReLU()
        )

        self.up1 = self._block(features*8 + 1024, features*8, down=False, act="relu", use_dropout=True)
        self.up2 = self._block(features*8*2, features*8, down=False, act="relu", use_dropout=True)
        self.up3 = self._block(features*8*2, features*8, down=False, act="relu", use_dropout=False)
        self.up4 = self._block(features*8*2, features*4, down=False, act="relu", use_dropout=False)
        self.up5 = self._block(features*4*2, features*2, down=False, act="relu", use_dropout=False)
        self.up6 = self._block(features*2*2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _block(
            self,
            in_channels: int, 
            out_channels: int, 
            down: bool = True, 
            act: str = "relu", 
            use_dropout: bool = False
        ) -> nn.Sequential:
        
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=="relu" else nn.LeakyReLU(0.2),
        )

        dropout = nn.Dropout(0.5)

        return nn.Sequential(conv, dropout) if use_dropout else conv
        
    
    def forward(
            self,
            x: torch.tensor
        ) -> torch.tensor:
        
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        
        bottleneck = self.bottleneck(d6)
        
        noise = torch.randn(bottleneck.shape[0], 1024).to(x.device)
        noise = noise.unsqueeze(2).unsqueeze(3)

        x_with_noise = torch.hstack([noise, bottleneck])

        up1 = self.up1(x_with_noise)
        up2 = self.up2(torch.cat([up1, d6], 1))
        up3 = self.up3(torch.cat([up2, d5], 1))
        up4 = self.up4(torch.cat([up3, d4], 1))
        up5 = self.up5(torch.cat([up4, d3], 1))
        up6 = self.up6(torch.cat([up5, d2], 1))
        
        return self.final_up(torch.cat([up6, d1],1))