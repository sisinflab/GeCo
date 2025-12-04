from torchvision.utils import save_image
from torch.utils.data import DataLoader
from custom_GAN.models.generator import Generator
import torch

@torch.no_grad
def save_some_examples(
        gen: Generator, 
        val_loader: DataLoader,
        epoch: int, 
        folder: str,
        device: torch.device
    ) -> None:
    
    batch = next(iter(val_loader))
    x = batch["tshirt_image"].to(device)
    y = batch["gt_image"].to(device)

    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()