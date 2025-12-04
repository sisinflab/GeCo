import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model.alexnet import AlexNetBackbone
from model.bpr_dae import BPRDAE
from utils.dataset import FashionDataset, FashionTaobaoTBDataset
from torchvision.utils import save_image
from itertools import product
import multiprocessing
import warnings
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
        img_encoder: AlexNetBackbone,
        bpr_net: BPRDAE,
        train_dl: DataLoader,
        val_dl: DataLoader,
        epoch: int,
        device: torch.device
    ) -> None:
    
    loop = tqdm(train_dl, desc='Training')
    val_loop = tqdm(val_dl, desc='Validation')
    running_loss, running_lrec, running_lbpr, running_l2_loss,  = [], [], [], []
    running_loss_val, running_lrec_val, running_lbpr_val, running_l2_loss_val = [], [], [], []

    bpr_net.train()

    for idx, batch in enumerate(loop):
        tshirts = batch['tshirt_image'].to(device)
        true_pants = batch['gt_image'].to(device)
        negative_pants = torch.stack([true_pants.roll(shifts=i + 1, dims=0) for i in range(3)], dim=1)

        tshirts = tshirts.unsqueeze(1).repeat_interleave(3, dim=1).to(device)
        true_pants = true_pants.unsqueeze(1).repeat_interleave(3, dim=1).to(device)

        b, r, c, w, h = tshirts.shape

        tshirts = tshirts.view(b*r, c, w, h)
        true_pants = true_pants.view(b*r, c, w, h)
        negative_pants = negative_pants.view(b*r, c, w, h)
        
        enc_top, enc_positive, enc_negative = img_encoder(tshirts, true_pants, negative_pants)
        cost, L_rec, L_bpr, L2_norm, _, _, _ = bpr_net.get_cost_updates(enc_top, enc_positive, enc_negative, backprop=True)

        running_loss.append(cost.item())
        running_lrec.append(L_rec)
        running_lbpr.append(L_bpr)
        running_l2_loss.append(L2_norm)


        logs = {
            "loss": np.mean(running_loss),
            "lrec": np.mean(running_lrec),
            "lbpr": np.mean(running_lbpr),
            "L2 norm": np.mean(running_l2_loss),
            "epoch": epoch
            }
        
        loop.set_postfix(**logs)

    bpr_net.eval()

    for idx, batch in enumerate(val_loop):
        tshirts = batch['tshirt_image'].unsqueeze(1).repeat_interleave(3, dim=1).to(device)
        true_pants = batch['gt_image'].unsqueeze(1).repeat_interleave(3, dim=1).to(device)
        negative_pants = batch['neg_images'].to(device)

        b, r, c, w, h = tshirts.shape

        tshirts = tshirts.view(b*r, c, w, h)
        true_pants = true_pants.view(b*r, c, w, h)
        negative_pants = negative_pants.view(b*r, c, w, h)
        
        enc_top, enc_positive, enc_negative = img_encoder(tshirts, true_pants, negative_pants)
        cost, L_rec, L_bpr, L2_norm, _, _, _ = bpr_net.get_cost_updates(enc_top, enc_positive, enc_negative, backprop=False)

        running_loss_val.append(cost.item())
        running_lrec_val.append(L_rec)
        running_lbpr_val.append(L_bpr)
        running_l2_loss_val.append(L2_norm)


        val_logs = {
            "loss": np.mean(running_loss_val),
            "lrec": np.mean(running_lrec_val),
            "lbpr": np.mean(running_lbpr_val),
            "L2 norm": np.mean(running_l2_loss_val),
            "epoch": epoch
            }
    
        val_loop.set_postfix(**val_logs)


def train(lr, hidden, lmb, mi, momentum, args):

    batch_size = args.batch_size
    valid_batch_size = args.valid_batch_size
    num_workers = args.num_workers
    img_size = args.img_size
    epochs = args.epochs
    dataset = args.dataset

    if not os.path.exists(args.weights_dir):
        os.mkdir(args.weights_dir)

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
        root_path = os.path.join(path, 'FashionTaobao-TB')
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


    img_encoder = AlexNetBackbone().to(device)
    bpr_net = BPRDAE(
        n_hidden_v=hidden,
        mu=mi,
        lmb=lmb,
        learning_rate=lr,
        momentum=momentum
    ).to(device)

    save_path = os.path.join(args.weights_dir, f'bprdae_{dataset}_{lr}_{hidden}_{lmb}_{mi}_{momentum}.pt') 

    for epoch in tqdm(range(epochs), desc='Epochs'):
        train_step(
            img_encoder=img_encoder,
            bpr_net=bpr_net,
            train_dl=train_dl,
            val_dl=val_dl,
            epoch=epoch,
            device=device
        )

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': bpr_net.state_dict(),
                'optimizer_state_dict': bpr_net.optimizer.state_dict(),
            },
            save_path
        )
    
    bpr_net.eval()

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
        auc_1vs3 = roc_auc(test_dl, img_encoder, bpr_net)
        auc_1vs1 = roc_auc_1vs1(test_dl, img_encoder, bpr_net)
        mrr_1vs9 = mrr(cir_dl, img_encoder, bpr_net)
        mrr_1vsall = mrr_1vsAll(cir_dl, img_encoder, bpr_net)

    del bpr_net, img_encoder
    torch.cuda.empty_cache()

    return auc_1vs1, mrr_1vsall, auc_1vs3, mrr_1vs9


@torch.no_grad()
def roc_auc(
    test_dl, 
    img_encoder, 
    bpr_net
) -> None:
    
    test_loop = tqdm(test_dl, desc='AUC computing', leave=False)
    running_auc_test = []

    for idx, batch in enumerate(test_loop):
        tshirts = batch['tshirt_image'].unsqueeze(1).repeat_interleave(3, dim=1).to(device)
        true_pants = batch['gt_image'].unsqueeze(1).repeat_interleave(3, dim=1).to(device)
        negative_pants = batch['neg_images'].to(device)

        b, r, c, w, h = tshirts.shape

        tshirts = tshirts.view(b*r, c, w, h)
        true_pants = true_pants.view(b*r, c, w, h)
        negative_pants = negative_pants.view(b*r, c, w, h)
        
        enc_top, enc_positive, enc_negative = img_encoder(tshirts, true_pants, negative_pants)

        emb_top, emb_pos, emb_neg = bpr_net.get_hidden_values(enc_top, enc_positive, enc_negative)

        mij = emb_top.mm(emb_pos.T).diag()
        mik = emb_top.mm(emb_neg.T).diag()
        
        score = torch.subtract(mij, mik)

        auc = (mij > mik).bool().sum().item() / (score.shape[0])
        running_auc_test.append(auc)

    return np.mean(running_auc_test)

@torch.no_grad()
def roc_auc_1vs1(
    test_dl, 
    img_encoder, 
    bpr_net
) -> np.array:
    
    test_loop = tqdm(test_dl, desc='AUC computing', leave=False)
    running_auc_test = []

    for batch in test_loop:
        tshirts = batch['tshirt_image'].to(device)
        true_pants = batch['gt_image'].to(device)
        negative_pants = batch['neg_images'].to(device)

        enc_top, enc_positive, enc_negative = img_encoder(tshirts, true_pants, negative_pants[:, 0])

        emb_top, emb_pos, emb_neg = bpr_net.get_hidden_values(enc_top, enc_positive, enc_negative)

        mij = emb_top.mm(emb_pos.T).diag()
        mik = emb_top.mm(emb_neg.T).diag()
        
        score = torch.subtract(mij, mik)

        auc = (mij > mik).bool().sum().item() / (score.shape[0])

        running_auc_test.append(auc)

    return np.mean(running_auc_test)

@torch.no_grad()
def mrr(
    cir_dl, 
    img_encoder, 
    bpr_net
) -> np.array:
    
    test_loop = tqdm(cir_dl, desc='MRR computing', leave=False)
    running_mrr = []

    for batch in test_loop:

        tshirts = batch['tshirt_image'].to(device)
        true_pants = batch['gt_image'].to(device)
        negative_pants = batch['neg_images'].to(device)

        enc_top, enc_positive, enc_negative = img_encoder(tshirts, true_pants, negative_pants.squeeze())
        emb_top, emb_pos, emb_neg = bpr_net.get_hidden_values(enc_top, enc_positive, enc_negative)
        enc_B_tot = torch.cat([emb_pos, emb_neg])

        similarity_matrix = emb_top.repeat(10, 1).mm(enc_B_tot.T).diag()
        _, indices = similarity_matrix.topk(k=10, largest=True)
        indices = list(indices)

        count = 1 / (indices.index(0) + 1)

        running_mrr.append(count)

    return np.mean(running_mrr)


@torch.no_grad()
def mrr_1vsAll(
    cir_dl,
    img_encoder, 
    bpr_net
) -> np.array:
    
    test_csv = cir_dl.dataset.data
    test_loop = tqdm(test_csv.iterrows(), desc='MRR computing 1vsAll', leave=False, total=len(test_csv))
    
    if isinstance(cir_dl.dataset, FashionTaobaoTBDataset):
        tshirt_dir = cir_dl.dataset.tshirt_dir
        pants_dir = cir_dl.dataset.pants_dir
    else:
        tshirt_dir = cir_dl.dataset.img_dir
        pants_dir = cir_dl.dataset.img_dir

    transforms_ = [
        transforms.Resize((224, 224), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]

    transform = transforms.Compose(transforms_)

    unique_pants = list(test_csv['positive_pant'].unique())

    negative_encs = []

    placeholder = torch.ones(1, 3, 224, 224, device=device)

    for neg in tqdm(unique_pants):
        negative_pant = transform(Image.open(os.path.join(pants_dir, str(neg) + '.jpg')).convert('RGB')).to(device)
        enc_top, enc_positive, enc_negative = img_encoder(placeholder, placeholder, negative_pant[None])
        _, _, emb_neg = bpr_net.get_hidden_values(enc_top, enc_positive, enc_negative)
        negative_encs.append(emb_neg.cpu().numpy())

    negative_encs = list(negative_encs)

    running_mrr = []

    for _, batch in test_loop:
        tshirt = transform(Image.open(os.path.join(tshirt_dir, str(batch['tshirt']) + '.jpg')).convert('RGB')).to(device)
        true_pant = transform(Image.open(os.path.join(pants_dir, str(batch['positive_pant']) + '.jpg')).convert('RGB')).to(device)

        enc_top, enc_positive, enc_negative = img_encoder(tshirt[None], true_pant[None], placeholder)

        emb_top, emb_pos, _ = bpr_net.get_hidden_values(enc_top, enc_positive, enc_negative)

        idx = unique_pants.index(batch['positive_pant'])
        
        actual_neg = torch.tensor(negative_encs[:idx] + negative_encs[idx+1:]).to(device)

        all_enc = torch.vstack([emb_pos[None], actual_neg]).squeeze()


        mij = emb_top.repeat(all_enc.shape[0], 1).mm(all_enc.T).diag()

        _, indices = mij.topk(k=all_enc.shape[0], largest=True)

        indices = list(indices)

        count = 1 / (indices.index(0) + 1)
        
        running_mrr.append(count)

    return np.mean(running_mrr)


if __name__ == '__main__':
    config = argparse.ArgumentParser(description="Training and evaluation script for BPRDAE.")
    config.add_argument('--dataset', choices=['fashionvc', 'expreduced', 'fashiontaobaoTB'], required=True)
    config.add_argument('--epochs', type=int, help="Number of epochs.", default=30)
    config.add_argument('--learning_rate_values', nargs='+', type=float, help='List of learning_rate values', default=0.01)
    config.add_argument('--n_hidden_values', nargs='+', type=int, help='List of n_hidden values', default=512)
    config.add_argument('--lmb_values', nargs='+', type=float, help='List of lmb values', default=0.01)
    config.add_argument('--mi_values', nargs='+', type=float, help='List of mi values', default=0.01)
    config.add_argument('--momentum_values', nargs='+', type=float, help='List of momentum values', default=0.9)
    config.add_argument('--device', type=str, help="device.", default='cuda')
    config.add_argument('--batch_size', type=int, help="Training Batch Size.", default=128)
    config.add_argument('--valid_batch_size', type=int, help="Valid Batch Size.", default=16)
    config.add_argument('--num_workers', type=int, help="Dataloader workers.", default=multiprocessing.cpu_count())
    config.add_argument('--img_size', type=int, help="Image Size.", default=224)
    config.add_argument('--out_csv', type=str, help="path of the output csv", default=os.path.join(os.getcwd(), 'baselines', 'BPRDAE', 'out.csv'))
    config.add_argument('--weights_dir', type=str, help="weights path", default=os.path.join(os.getcwd(), 'baselines', 'BPRDAE', 'weights'))

    args = config.parse_args()

    lrs = args.alpha_values if type(args.learning_rate_values) is list else [args.learning_rate_values]
    hiddens = args.beta_values if type(args.n_hidden_values) is list else [args.n_hidden_values]
    lmbds = args.mi_values if type(args.lmb_values) is list else [args.lmb_values]
    mis = args.ni_values if type(args.mi_values) is list else [args.mi_values]
    momentums = args.ni_values if type(args.momentum_values) is list else [args.momentum_values]

    hyperparameter_grid = list(product(lrs, hiddens, lmbds, mis, momentums))
    hyperparameter_grid = [quadruple for quadruple in hyperparameter_grid]

    if os.path.exists(args.out_csv):
        raise ValueError("Cannot overwrite existing file!")

    with open(args.out_csv, "a") as log_file:
        log_file.write("learning_rate,n_hidden,lambda,mi,momentum,auc,mrr,auc_baseline,mrr_baseline\n")
    log_file.close()

    for lr, hidden, lmb, mi, momentum in tqdm(hyperparameter_grid, desc='GRID', leave=False):
        with open(args.out_csv, "a") as log_file:
            auc_1vs1, mrr_1vsall, auc_1vs3, mrr_1vs9 = train(lr, hidden, lmb, mi, momentum, args)
            log_file.write(f"{lr},{hidden},{lmb},{mi},{momentum},{auc_1vs1},{mrr_1vsall},{auc_1vs3},{mrr_1vs9}\n")
        log_file.close()
        torch.cuda.empty_cache()
        
