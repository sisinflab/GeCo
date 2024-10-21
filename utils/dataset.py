import os
import torch
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
import multiprocessing


class FashionTaobaoTBDataset(Dataset):
    def __init__(
            self,
            root_path: str,
            transforms_: list = None,
            mode: str = "train",
            img_size: int = 64,
            ) -> None:
        
        super(FashionTaobaoTBDataset, self).__init__()

        self.transform = transforms.Compose(transforms_)
        self.mode = mode
        self.img_size = img_size

        if mode == 'train':
            f = os.path.join(root_path, os.path.join('files', 'train_no_dup_1vs9.csv'))
            self.data = pd.read_csv(f)

        elif mode == 'valid':
            f = os.path.join(root_path, os.path.join('files', 'valid_no_dup_1vs9.csv'))
            self.data = pd.read_csv(f)
            
        elif mode == 'test' or mode == 'CIR':
            f = os.path.join(root_path, os.path.join('files', 'test_disj.csv'))
            self.data = pd.read_csv(f)

        else:
            raise ValueError("mode should be 'train', 'valid' or 'test'")

        self.img_dir = os.path.join(root_path, 'img')

    def __getitem__(
            self,
            index: int
            ) -> dict:
        
        data = self.data.loc[index]

        tshirt_image = self.transform(Image.open(os.path.join(self.img_dir, str(data['tshirt']) + '.jpg')).convert('RGB'))
        gt_pant = self.transform(Image.open(os.path.join(self.img_dir, str(data['positive_pant']) + '.jpg')).convert('RGB'))

        if not self.mode == 'CIR':
            neg_imgs = torch.empty(3, 3, self.img_size, self.img_size)

            for i in range(len(neg_imgs)):
                neg_imgs[i] = self.transform(Image.open(os.path.join(self.img_dir, str(data[f'neg_{i+1}']) + '.jpg')).convert('RGB'))
            
            dic = {
                'tshirt_image': tshirt_image,
                'gt_image': gt_pant,
                'neg_images': neg_imgs,
                'tshirt_id': data['tshirt'],
                'pant_id': data['positive_pant'],
                'neg_1_id': data['neg_1'],
                'neg_2_id': data['neg_2'],
                'neg_3_id': data['neg_3'],
                }
            
        else:
            neg_imgs = torch.empty(9, 3, self.img_size, self.img_size)

            for i in range(len(neg_imgs)):
                neg_imgs[i] = self.transform(Image.open(os.path.join(self.img_dir, str(data[f'neg_{i+1}']) + '.jpg')).convert('RGB'))
            
            dic = {
                'tshirt_image': tshirt_image,
                'gt_image': gt_pant,
                'neg_images': neg_imgs,
                'tshirt_id': data['tshirt'],
                'pant_id': data['positive_pant'],
                'neg_1_id': data['neg_1'],
                'neg_2_id': data['neg_2'],
                'neg_3_id': data['neg_3'],
                'neg_4_id': data['neg_4'],
                'neg_5_id': data['neg_5'],
                'neg_6_id': data['neg_6'],
                'neg_7_id': data['neg_7'],
                'neg_8_id': data['neg_8'],
                'neg_9_id': data['neg_9'],
                }

        return dic

    def __len__(self) -> int:
        return len(self.data)


class FashionDataset(Dataset):
    def __init__(
            self,
            root_path: str,
            transforms_: list = None,
            mode: str = "train",
            img_size: int = 64,
            name: str = None
        ) -> None:

        super(FashionDataset, self).__init__()

        self.transform = transforms.Compose(transforms_)
        self.mode = mode
        self.img_size = img_size

        if not name == 'expreduced':
            train_file = os.path.join('files', 'traindata_unique.csv')
            valid_file = os.path.join('files', 'devdata_tops.csv')
            test_file = os.path.join('files', 'testdata_tops.csv')
        else:
            train_file = os.path.join('files', 'traindata_unique_1.csv')
            valid_file = os.path.join('files', 'devdata_tops_reduced_rs1.csv')
            test_file = os.path.join('files', 'testdata_tops_reduced_rs1.csv')

        if mode == 'train':
            f = os.path.join(root_path, train_file)
            self.data = pd.read_csv(f)

        elif mode == 'valid':
            f = os.path.join(root_path, valid_file)
            self.data = pd.read_csv(f)

        elif mode == 'test' or mode == 'CIR':
            f = os.path.join(root_path, test_file)
            self.data = pd.read_csv(f)

        else:
            raise ValueError("mode should be 'train', 'valid' or 'test'")

        self.img_dir = os.path.join(root_path, 'img')

    def __getitem__(
            self,
            index: int
    ) -> dict:

        data = self.data.loc[index]

        tshirt_image = self.transform(
            Image.open(os.path.join(self.img_dir, str(data['tshirt']) + '.jpg')).convert('RGB'))
        gt_pant = self.transform(
            Image.open(os.path.join(self.img_dir, str(data['positive_pant']) + '.jpg')).convert('RGB'))

        if self.mode == 'train':
            dic = {
                'tshirt_image': tshirt_image,
                'gt_image': gt_pant,
                'tshirt_id': data['tshirt'],
                'pant_id': data['positive_pant'],
            }

        elif not self.mode == 'CIR':
            neg_imgs = torch.empty(3, 3, self.img_size, self.img_size)

            for i in range(len(neg_imgs)):
                neg_imgs[i] = self.transform(
                    Image.open(os.path.join(self.img_dir, str(data[f'neg_{i + 1}']) + '.jpg')).convert('RGB'))

            dic = {
                'tshirt_image': tshirt_image,
                'gt_image': gt_pant,
                'neg_images': neg_imgs,
                'tshirt_id': data['tshirt'],
                'pant_id': data['positive_pant'],
                'neg_1_id': data['neg_1'],
                'neg_2_id': data['neg_2'],
                'neg_3_id': data['neg_3'],
            }

        else:
            neg_imgs = torch.empty(9, 3, self.img_size, self.img_size)

            for i in range(len(neg_imgs)):
                neg_imgs[i] = self.transform(
                    Image.open(os.path.join(self.img_dir, str(data[f'neg_{i + 1}']) + '.jpg')).convert('RGB'))

            dic = {
                'tshirt_image': tshirt_image,
                'gt_image': gt_pant,
                'neg_images': neg_imgs,
                'tshirt_id': data['tshirt'],
                'pant_id': data['positive_pant'],
                'neg_1_id': data['neg_1'],
                'neg_2_id': data['neg_2'],
                'neg_3_id': data['neg_3'],
                'neg_4_id': data['neg_4'],
                'neg_5_id': data['neg_5'],
                'neg_6_id': data['neg_6'],
                'neg_7_id': data['neg_7'],
                'neg_8_id': data['neg_8'],
                'neg_9_id': data['neg_9'],
            }

        return dic

    def __len__(self) -> int:
        return len(self.data)


if __name__ == '__main__':

    config = argparse.ArgumentParser(description="Dataset Test.")
    config.add_argument('--img_size', type=int, help="Dimension of the image", default=128)
    config.add_argument('--dataset', choices=['fashionvc', 'expreduced', 'fashiontaobaoTB'])
    config.add_argument('--batch_size', type=int, help="batch size", default=64)
    config.add_argument('--num_workers', type=int, help="num workers", default=multiprocessing.cpu_count())
    config.add_argument('--mode', choices=['train', 'valid', 'test'])
    
    args = config.parse_args()

    transforms_ = [
        transforms.Resize((args.img_size, args.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    path = os.path.join(os.getcwd(), 'datasets')
    if args.dataset == 'fashionvc':
        path = os.path.join(path, 'FashionVC')
        dataset = FashionDataset(
            root_path=path,
            mode=args.mode,
            transforms_=transforms_,
            img_size=args.img_size
            )
    elif args.dataset == 'expreduced':
        path = os.path.join(path, 'ExpReduced')
        dataset = FashionDataset(
            root_path=path,
            mode=args.mode,
            transforms_=transforms_,
            img_size=args.img_size
            )
    elif args.dataset == 'fashiontaobaoTB':
        path = os.path.join(path, 'FashionTaobao-TB')
        dataset = FashionTaobaoTBDataset(
            root_path=path,
            mode=args.mode,
            transforms_=transforms_,
            img_size=args.img_size
            )
    else:
        raise ValueError("Dataset must be in ['fashionvc', 'expreduced', 'fashiontaobaoTB']")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    for t in dataloader:
        continue

    print('SUCCESS')