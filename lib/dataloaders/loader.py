import random
import numpy as np
from PIL import Image
import os.path as osp
from torch.utils import data
from torchvision import transforms

def loadTxt(txt_path):

    img_path = []
    labels = []

    with open(txt_path,'r') as data_file:
        for line in data_file:
            data = line.split()
            img_path.append(data[0])
            labels.append(data[1])

    return img_path, np.asarray(labels, dtype=np.int64)

class Dataset(data.Dataset):
    def __init__(self, root_path, list_path, transform=None):
        self.root = root_path
        self.list_path = list_path
        self.transform = transform
        self.imgs, self.labels = loadTxt(list_path)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]

        with open(osp.join(self.root, img_path), "rb") as f:
             sample = Image.open(f)
             sample = sample.convert("RGB")

        target = self.labels[index]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target
    
def createLoader(num_workers, input_size, data_dir, list_path, train, batch_size):
    ### Load dataset
    h, w = map(int, input_size.split(','))
    input_size = (h, w)


    transforms_ = transforms.Compose([transforms.Resize(input_size),
                                     transforms.ToTensor(),])

    dataset = Dataset(root_path=data_dir,
                        list_path=list_path, 
                        transform=transforms_)

    loader = data.DataLoader(dataset,
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=num_workers, 
                                pin_memory=True,
                                drop_last = True if train else False)

    return loader

