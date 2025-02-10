import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from util.logconf import logging
import util.augmentation as augmentation

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


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

        log.info("{} {} samples".format( len(self.datalines), data_type))

    def __len__(self):
        return len(self.datalines)

    def __getitem__(self, idx):
        dataline = self.datalines[idx]

        label_path = os.path.join(self.root, dataline)
        img_path = label_path.replace("label", "image")

        image = Image.open(img_path)
        label = Image.open(label_path).convert('L')

        # TODO: 后续将数据增强和数据转换分开，因为数据增强可以使用tensor类型并在GPU上运行，而数据转换只能使用PIL类型或者numpy类型
        # 设置transform，数据预处理
        image_transform = augmentation.get_transform(image.size, IsResize=True, IsTotensor=True, IsNormalize=True)
        label_transform = augmentation.get_transform(image.size, IsResize=True, IsTotensor=False, IsNormalize=False)


        # PIL读取的图片默认是 H W C 形状，需要转换成 C H W, 并且将像素值从[0,255]转换成[0,1]
        # image = torch.from_numpy(np.array(image) / 255.0).permute(2, 0, 1).to(torch.float32)

        # ToTensor能够将PIL和numpy格式的图片的数值范围从[0,255]->[0,1],且将图像形状从[H,W,C]->[C,H,W]
        image = image_transform(image)

        label = label_transform(label)
        # label不能用ToTensor进行转换，因为label需要保持为整数才能做真值运算，而ToTensor会将其转换成浮点数
        # 标签形状为[H,W]，不需要通道
        label = torch.from_numpy(np.array(label))

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
