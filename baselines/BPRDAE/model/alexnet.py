import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple

class AlexNetBackbone(nn.Module):
    def __init__(
            self
        ) -> None:
        
        super(AlexNetBackbone, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        self.features = alexnet.features
        self.avgpool = alexnet.avgpool
        self.classifier = nn.Sequential(*list(alexnet.classifier.children())[:-1])  # Exclude the last fully connected layer
        self.features.eval()
        self.avgpool.eval()
        self.classifier.eval()

    def forward_once(
            self,
            x: torch.Tensor
        ) -> torch.Tensor:
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def forward(
            self,
            top: torch.Tensor,
            pos_bottom: torch.Tensor,
            neg_bottom: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        top_enc = self.forward_once(top)
        pos_enc = self.forward_once(pos_bottom)
        neg_enc = self.forward_once(neg_bottom)
        
        return top_enc, pos_enc, neg_enc
                

if __name__ == '__main__':
    alexnet = AlexNetBackbone()
    _, _, _ = alexnet(torch.randn(1, 3, 224, 224),torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224))

    print('SUCCESS')