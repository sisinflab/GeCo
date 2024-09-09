import os
import random
from tqdm import tqdm
from model import CGCRM
from utils.dataset import FashionDataset, FashionTaobaoTBDataset
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from CIGM.models.generator import Generator
from itertools import product
import warnings
import argparse
import multiprocessing
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True

def train_step(
        model: CGCRM,
        generator: Generator,
        train_dl: DataLoader,
        val_dl: DataLoader,
        epoch: int,
) -> None:

    loop = tqdm(train_dl, desc='Training')
    val_loop = tqdm(val_dl, desc='Validation')
    running_loss, running_auc = [], []
    running_loss_val, running_auc_val = [], []

    model.train()

    for idx, batch in enumerate(loop):
        tshirts = batch['tshirt_image'].to(device)
        true_pants = batch['gt_image'].to(device)
        negative_pants = torch.stack([true_pants.roll(shifts=1, dims=0)], dim=1).squeeze()

        with torch.no_grad():
            template = generator(tshirts)

        cost, pos_scores, neg_scores = model.get_cost_updates(
            tshirts, 
            true_pants, 
            negative_pants, 
            template,
            backprop=True
        )

        running_loss.append(cost.item())

        auc = (pos_scores > neg_scores).bool().sum().item()

        auc = auc / (tshirts.shape[0])
        running_auc.append(auc)

        logs = {
            "loss": np.mean(running_loss),
            "auc": np.mean(running_auc),
            "epoch": epoch
        }

        loop.set_postfix(**logs)

    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(val_loop):
            tshirts = batch['tshirt_image'].unsqueeze(1).repeat_interleave(3, dim=1).to(device)
            true_pants = batch['gt_image'].unsqueeze(1).repeat_interleave(3, dim=1).to(device)
            negative_pants = batch['neg_images'].to(device)

            b, r, c, w, h = tshirts.shape

            tshirts = tshirts.view(b * r, c, w, h)
            true_pants = true_pants.view(b * r, c, w, h)
            negative_pants = negative_pants.view(b * r, c, w, h)

            template = generator(tshirts)

            cost, pos_scores, neg_scores = model.get_cost_updates(
                tshirts, 
                true_pants, 
                negative_pants, 
                template,
                backprop=False
            )

            running_loss_val.append(cost.item())

            auc = (pos_scores > neg_scores).bool().sum().item()

            auc = auc / (tshirts.shape[0])
            running_auc_val.append(auc)

            val_logs = {
                "loss": np.mean(running_loss_val),
                "auc": np.mean(running_auc_val),
                "epoch": epoch
            }

            val_loop.set_postfix(**val_logs)

    return np.mean(running_auc_val)


def train(
        alpha, 
        beta, 
        gamma, 
        temperature, 
        args
    ):

    batch_size = args.train_batch_size
    valid_batch_size = args.valid_batch_size
    num_workers = args.num_workers
    emb_dim = args.emb_dim
    learning_rate = args.learning_rate
    img_size = args.img_size
    epochs = args.num_epochs
    dataset = args.dataset
    generator_path = os.path.join(args.generator_dir, f'generator_{dataset}.pt')
    

    if not os.path.exists(args.weights_dir):
        os.mkdir(args.weights_dir)

    weight_path = os.path.join(args.weights_dir,
                               f"cgcrm_{dataset}_{epochs}_{alpha}_{beta}_{gamma}_{temperature}.pt")

    cgcrm = CGCRM(
        emb_dim=emb_dim,
        learning_rate=learning_rate,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        temperature=temperature
    ).to(device)

    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device)['model_state_dict'])
    generator.eval()

    transforms_ = [
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]

    path = os.path.join(os.getcwd(), 'datasets')

    if args.dataset == 'fashionvc':
        root_path = os.path.join(path, 'FashionVC')
        train_dataset = FashionDataset(root_path, mode='train', transforms_=transforms_, img_size=img_size, name=dataset)
        valid_dataset = FashionDataset(root_path, mode='valid', transforms_=transforms_, img_size=img_size, name=dataset)
        test_dataset = FashionDataset(root_path, mode='test', transforms_=transforms_, img_size=img_size, name=dataset)
        cir_dataset = FashionDataset(root_path, mode='CIR', transforms_=transforms_, img_size=img_size, name=dataset)

    elif dataset == 'expreduced':
        root_path = os.path.join(path, 'ExpReduced')
        train_dataset = FashionDataset(root_path, mode='train', transforms_=transforms_, img_size=img_size, name=dataset)
        valid_dataset = FashionDataset(root_path, mode='valid', transforms_=transforms_, img_size=img_size, name=dataset)
        test_dataset = FashionDataset(root_path, mode='test', transforms_=transforms_, img_size=img_size, name=dataset)
        cir_dataset = FashionDataset(root_path, mode='CIR', transforms_=transforms_, img_size=img_size, name=dataset)

    else:
        root_path = os.path.join(path, 'ExpReduced')
        train_dataset = FashionTaobaoTBDataset(root_path, mode='train', transforms_=transforms_, img_size=img_size)
        valid_dataset = FashionTaobaoTBDataset(root_path, mode='valid', transforms_=transforms_, img_size=img_size)
        test_dataset = FashionTaobaoTBDataset(root_path, mode='test', transforms_=transforms_, img_size=img_size)
        cir_dataset = FashionTaobaoTBDataset(root_path, mode='CIR', transforms_=transforms_, img_size=img_size)

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_dl = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    best_auc_val = 0

    for epoch in tqdm(range(epochs), desc='Epochs'):
        auc_val = train_step(
            model=cgcrm,
            generator=generator,
            train_dl=train_dl,
            val_dl=val_dl,
            epoch=epoch,
        )

        cgcrm.scheduler.step()

        if auc_val > best_auc_val:
            best_auc_val = auc_val
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': cgcrm.state_dict(),
                    'optimizer_state_dict': cgcrm.optimizer.state_dict(),
                },
                weight_path
            )
    
    del generator, cgcrm
    torch.cuda.empty_cache()

    cgcrm = CGCRM(
        emb_dim=emb_dim,
        learning_rate=learning_rate,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        temperature=temperature
    ).to(device)

    cgcrm.load_state_dict(torch.load(weight_path, map_location=device)['model_state_dict'])
    cgcrm.eval()

    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device)['model_state_dict'])
    generator.eval()

    test_dl = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    cir_dl = DataLoader(
            cir_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    with torch.no_grad():
        auc_1vs3 = roc_auc(test_dl, generator, cgcrm)
        auc_1vs1 = roc_auc_1vs1(test_dl, generator, cgcrm)
        mrr_1vs9 = mrr(cir_dl, generator, cgcrm)
        mrr_1vsall = mrr_1vsAll(cir_dl, generator, cgcrm)

    return auc_1vs3, auc_1vs1, mrr_1vs9, mrr_1vsall, weight_path


@torch.no_grad()
def roc_auc(
    test_dl, 
    generator, 
    compatibility_network
) -> np.array:
    
    test_loop = tqdm(test_dl, desc='AUC computing', leave=False)
    running_auc_test = []

    for batch in test_loop:
        tshirts = batch['tshirt_image'].unsqueeze(1).repeat_interleave(3, dim=1).to(device)
        true_pants = batch['gt_image'].unsqueeze(1).repeat_interleave(3, dim=1).to(device)
        negative_pants = batch['neg_images'].to(device)

        b, r, c, w, h = tshirts.shape

        tshirts = tshirts.view(b * r, c, w, h)
        true_pants = true_pants.view(b * r, c, w, h)
        negative_pants = negative_pants.view(b * r, c, w, h)

        template = generator(tshirts)

        _, pos_scores, neg_scores = compatibility_network.get_cost_updates(tshirts, true_pants, negative_pants,
                                                                                template, backprop=False)

        auc = (pos_scores > neg_scores).bool().sum().item()

        auc = auc / (tshirts.shape[0])
        running_auc_test.append(auc)

    return np.mean(running_auc_test)

@torch.no_grad()
def roc_auc_1vs1(
    test_dl, 
    generator, 
    compatibility_network
) -> np.array:
    
    test_loop = tqdm(test_dl, desc='AUC computing', leave=False)
    running_auc_test = []

    for batch in test_loop:
        tshirts = batch['tshirt_image'].to(device)
        true_pants = batch['gt_image'].to(device)
        negative_pants = batch['neg_images'][:, 0].to(device)
        template = generator(tshirts)

        _, pos_scores, neg_scores = compatibility_network.get_cost_updates(tshirts, true_pants, negative_pants,
                                                                                template, backprop=False)

        auc = (pos_scores > neg_scores).bool().sum().item()

        auc = auc / (tshirts.shape[0])
        running_auc_test.append(auc)

    return np.mean(running_auc_test)

@torch.no_grad()
def mrr(
    cir_dl, 
    generator, 
    compatibility_network
) -> np.array:
    
    test_loop = tqdm(cir_dl, desc='MRR computing', leave=False)
    running_mrr = []

    for batch in test_loop:
        tshirts = batch['tshirt_image'].to(device)
        true_pants = batch['gt_image'].to(device)
        negative_pants = batch['neg_images'].to(device)

        all_imgs = torch.vstack([true_pants, negative_pants.squeeze()])
        query_img = generator(tshirts)

        query_enc = compatibility_network.forward_query(tshirts, query_img)
        all_enc = compatibility_network.forward_bottom(all_imgs)

        similarity_matrix = torch.mul(query_enc, all_enc).sum(dim=1)

        _, indices = similarity_matrix.topk(k=10, largest=True)

        indices = list(indices)

        count = 1 / (indices.index(0) + 1)

        running_mrr.append(count)

    return np.mean(running_mrr)

@torch.no_grad()
def mrr_1vsAll(
    cir_dl, 
    generator, 
    compatibility_network
) -> np.array:
    
    test_csv = cir_dl.dataset.data
    test_loop = tqdm(test_csv.iterrows(), desc='MRR computing 1vsAll', leave=False, total=len(test_csv))

    if isinstance(cir_dl.dataset, FashionTaobaoTBDataset):
        tshirt_dir = cir_dl.dataset.tshirt_dir
        pants_dir = cir_dl.dataset.pants_dir
    else:
        tshirt_dir = cir_dl.dataset.img_dir
        pants_dir = cir_dl.dataset.img_dir

    transform = transforms.Compose([
            transforms.Resize((128, 128), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    unique_pants = list(test_csv['positive_pant'].unique())

    negative_encs = []

    for neg in tqdm(unique_pants):
        negative_pant = transform(Image.open(os.path.join(pants_dir, str(neg) + '.jpg')).convert('RGB')).to(
            device)
        negative_encs.append(compatibility_network.forward_bottom(negative_pant[None]).cpu().numpy())

    negative_encs = list(negative_encs)

    running_mrr = []

    for _, batch in test_loop:
        tshirt = transform(Image.open(os.path.join(tshirt_dir, str(batch['tshirt']) + '.jpg')).convert('RGB')).to(
            device)
        fake = generator(tshirt[None])
        query_enc = compatibility_network.forward_query(tshirt[None], fake)

        true_pant = transform(
            Image.open(os.path.join(pants_dir, str(batch['positive_pant']) + '.jpg')).convert('RGB')).to(
            device)
        true_enc = compatibility_network.forward_bottom(true_pant[None])
        idx = unique_pants.index(batch['positive_pant'])

        actual_neg = torch.tensor(negative_encs[:idx] + negative_encs[idx + 1:]).to(device)

        all_enc = torch.vstack([true_enc[None], actual_neg]).squeeze()

        similarity_matrix = torch.mul(query_enc, all_enc).sum(dim=1)

        _, indices = similarity_matrix.topk(k=all_enc.shape[0], largest=True)

        indices = list(indices)

        count = 1 / (indices.index(0) + 1)

        running_mrr.append(count)
    return np.mean(running_mrr)


# CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. python3 ProposedModels/unified_query_tests_traindata/exploration_traindata.py


if __name__ == '__main__':
    config = argparse.ArgumentParser(description="Training and evaluation script for CGCRM.")
    config.add_argument('--alpha_values', nargs='+', type=float, help='List of alpha values', required=True)
    config.add_argument('--beta_values', nargs='+', type=float, help='List of beta values', required=True)
    config.add_argument('--gamma_values', nargs='+', type=float, help='List of gamma values', required=True)
    config.add_argument('--tau_values', nargs='+', type=float, help='List of tau values', required=True)
    config.add_argument('--dataset', choices=['fashionvc', 'expreduced', 'fashiontaobaoTB'], required=True)
    config.add_argument('--num_epochs', type=int, help="Number of epochs.", default=50)
    config.add_argument('--emb_dim', type=int, help="Embedding dimension.", default=128)
    config.add_argument('--beta1', type=float, help="Adam Beta1.", default=0.5)
    config.add_argument('--learning_rate', type=float, help="Learning rate", default=1e-4)
    config.add_argument('--device', type=str, help="device", default='cuda')
    config.add_argument('--img_size', type=int, help="image size", default=128)
    config.add_argument('--train_batch_size', type=int, help="train batch size", default=64)
    config.add_argument('--valid_batch_size', type=int, help="valid batch size", default=16)
    config.add_argument('--num_workers', type=int, help="num workers", default=multiprocessing.cpu_count())
    config.add_argument('--weights_dir', type=str, help="weights path", default=os.path.join(os.getcwd(), 'CGCRM', 'weights'))
    config.add_argument('--generator_dir', type=str, help="generator path", default=os.path.join(os.getcwd(), 'CIGM', 'weights'))
    config.add_argument('--out_csv', type=str, help="path of the output csv", default=os.path.join(os.getcwd(), 'CGCRM', 'out.csv'))

    args = config.parse_args()
    alphas = args.alpha_values
    betas = args.beta_values
    gammas = args.gamma_values
    taus = args.tau_values
    hyperparameter_grid = list(product(alphas, betas, gammas, taus))
    hyperparameter_grid = [quadruple for quadruple in hyperparameter_grid if not (quadruple[0] == 0 and quadruple[1] == 0)]

    if os.path.exists(args.out_csv):
        raise ValueError("Cannot overwrite existing file!")

    with open(args.out_csv, "a") as log_file:
        log_file.write("alpha,beta,gamma,tau,auc_1vs3,auc_1vs1,mrr_1vs3,mrr_1vsall,weight_path\n")
    log_file.close()

    for a, b, g, t in tqdm(hyperparameter_grid, desc='GRID', leave=False):
        with open(args.out_csv, "a") as log_file:
            auc_1vs3, auc_1vs1, mrr_1vs9, mrr_1vs_all, weight_path = train(a, b, g, t, args)
            log_file.write(f"{a},{b},{g},{t},{auc_1vs3},{auc_1vs1},{mrr_1vs9},{mrr_1vs_all},{weight_path}\n")
        log_file.close()
        torch.cuda.empty_cache()
        
