import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image

class DaedalusDataset(Dataset):
    def __init__(self, dataset_path, transform=ToTensor()):
        self.dataset_path = dataset_path

        self.hr_img_name_list = os.listdir(
            os.path.join(self.dataset_path, 'image-HR'))
        self.hr_img_name_list.sort()

        self.lr_img_name_list = os.listdir(
            os.path.join(self.dataset_path, 'image-LR'))
        self.lr_img_name_list.sort()

        self.label_name_list = os.listdir(
            os.path.join(self.dataset_path, 'mask'))
        self.label_name_list.sort()

        self.transform = transform

    def __len__(self):
        return len(self.hr_img_name_list)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.dataset_path, 'image-HR', self.hr_img_name_list[idx])
        hr_img = self.transform(Image.open(hr_path))

        lr_path = os.path.join(self.dataset_path, 'image-LR', self.lr_img_name_list[idx])
        lr_img = self.transform(Image.open(lr_path))

        img = torch.stack((hr_img.squeeze(dim=0), lr_img.squeeze(dim=0)))

        mask_path = os.path.join(self.dataset_path, 'mask', self.label_name_list[idx])
        mask = self.transform(Image.open(mask_path)).squeeze(dim=0)

        return {'idx': torch.tensor(idx), 'image': img, 'mask': mask}


if __name__ == '__main__':
    from torchvision import transforms

    dataset = DaedalusDataset(
        '/home/supreme/datasets-nas/INAF/daedalus/dataset-pairs/train',
        transform=transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
    )
    sample = dataset.__getitem__(0)
    print(sample['image'].shape, sample['mask'].shape)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=1
    )
    for i, (idxs, images, labels) in enumerate(dataloader):
        print(f'testing batches {i}\r', end="")


