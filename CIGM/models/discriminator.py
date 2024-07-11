import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            features: list = [64, 128, 256, 512]
        ) -> None:
        
        super(Discriminator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]

        for feature in features[1:]:
            layers.append(
                self._block(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )

            in_channels = feature
            
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=0)
            )

        self.model = nn.Sequential(*layers)

    def _block(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 2
        ) -> nn.Sequential:
        
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect"),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )

    def forward(
            self, 
            x: torch.tensor, 
            y: torch.tensor
        ) -> torch.tensor:
        
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x
    
