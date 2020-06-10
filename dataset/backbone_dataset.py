import os

import torch
from torch.utils.data.dataset import Dataset
import  torchvision.transforms as T
from PIL import Image


class ImageNetDataset(Dataset):
    """
    ImageNet Dataset used while training Resnet backbone
    """

    def __init__(self, root, transform=None,  mode='train'):
        super(ImageNetDataset, self).__init__()
        self.root = root
        self.transform = transform

        with open(os.path.join(self.root, mode+".txt"), 'r') as f:
            raw_data = f.read().split()
            self.file_list = raw_data[::2]
            self.gt_list = raw_data[1::2]
        
        self.file_list = [os.path.join(os.path.join(root,mode),file) for file in self.file_list]
        self.gt_list = [int(gt) for gt in self.gt_list]

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx])
        if img.mode != "RGB":
            # print("bad image", self.file_list[idx])
            # img = Image.open(self.file_list[0]) 
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        
        return img, self.gt_list[idx]


if __name__ == '__main__':
    import torch.utils.data as D
    from transform import RandomResize 
    tran = T.Compose([
        RandomResize([256, 480]),
        T.RandomHorizontalFlip(),
        T.RandomCrop(224),
        T.ToTensor(), 
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    data = ImageNetDataset('/share/data/zhangyabo2/ImageNet/', transform=tran, mode='val')
    loader = D.DataLoader(data, batch_size=2)
    for img, gt in loader:
        print(img.shape)
        print(gt)
        break
