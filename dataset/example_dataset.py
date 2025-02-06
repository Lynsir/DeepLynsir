import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class ExampleDataset(Dataset):
    def __init__(self, data_type='trn'):
        self.root = r"F:\Projects\UNet"
        if data_type.lower() == 'trn':
            self.source = os.path.join(self.root, "data/list/train.txt")
        elif data_type.lower() == 'val':
            self.source = os.path.join(self.root, "data/list/val.txt")
        elif data_type.lower() == 'tst':
            self.source = os.path.join(self.root, "data/list/test.txt")

        with open(self.source, "r") as f:
            # 从文件中读取行默认结尾有一个'\n', 直接切片或者split都可以
            self.datalines = [dt[:-1] for dt in f.readlines()]


    def __len__(self):
        return len(self.datalines)


    def __getitem__(self, idx):
        dataline = self.datalines[idx]

        label_path = os.path.join(self.root, dataline)
        img_path = label_path.replace("label", "image")

        image = Image.open(img_path)
        label = Image.open(label_path)

        # PIL读取的图片默认是 H W C 形状，需要转换成 C H W
        image = torch.from_numpy(np.array(image)/255.0).permute(2, 0, 1).to(torch.float32)
        label = torch.from_numpy(np.array(label)[:,:,0]).unsqueeze(0)

        return image, label


if __name__ == '__main__':
    print("This is test for example_dataset.py")
    dataset = ExampleDataset(data_type='trn')
    img, lab = dataset[0]
    print(lab.shape)
    print(lab)
    import matplotlib.pyplot as plt
    lab_a = lab.numpy()
    plt.imshow(lab_a[0], cmap='gray')
    plt.show()



