import torch
from torch.utils.data import Dataset
import os
import numpy as np
import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from PIL import Image

from util import augmentation
from util.logconf import logging
log = logging.getLogger(__name__)


class LGGSegDataset(Dataset):
    def __init__(self, data_type='trn', is_create=False):
        self.root = r"F:\Projects\Datasets\Brain MRI segmentation (LGG Segmentation Dataset) by Mateusz Buda"

        self.__class__.createLGGSegDatasetFiles(self.root) if is_create else None

        if data_type.lower() == 'trn':
            self.source = os.path.join(self.root, "trn.txt")
        elif data_type.lower() == 'val':
            self.source = os.path.join(self.root, "val.txt")

        with open(self.source, "r") as f:
            # 从文件中读取行默认结尾有一个'\n'
            self.datalines = [dt.strip() for dt in f.readlines()]

        log.info("{} {} samples".format( len(self.datalines), data_type))

    def __len__(self):
        return len(self.datalines)

    def __getitem__(self, idx):
        dataline = self.datalines[idx]

        image = Image.open(dataline.replace("_mask", ""))
        label = Image.open(dataline).convert('L')

        # 设置transform，数据预处理
        image_transform = augmentation.get_transform(image.size, IsResize=True, IsTotensor=True, IsNormalize=True)
        label_transform = augmentation.get_transform(image.size, IsResize=True, IsTotensor=False, IsNormalize=False)

        image = image_transform(image)
        label = label_transform(label)

        label = torch.from_numpy(np.array(label)//255)

        return image, label


    @staticmethod
    def createLGGSegDatasetFiles(root):
        """
        生成数据集文件
        :param root: 文件夹路径
        """
        if os.path.exists(os.path.join(root, "trn.txt")) and os.path.exists(os.path.join(root, "val.txt")):
            print("Dataset files already exist.")
            return

        patient_list = glob(os.path.join(root, r"lgg-mri-segmentation\*"))
        patient_list = [p for p in patient_list if os.path.isdir(p)]

        trn_dirlist, val_dirlist = train_test_split(patient_list, test_size=0.2, random_state=42)

        trn_list = []
        val_list = []
        for p in trn_dirlist:
            trn_list += glob(os.path.join(p, r"*mask*.tif"))

        for p in val_dirlist:
            val_list += glob(os.path.join(p, r"*mask*.tif"))

        with open(os.path.join(root, "trn.txt"), "w") as f:
            for p in tqdm:
                f.write(p + "\n")

        with open(os.path.join(root, "val.txt"), "w") as f:
            for p in val_list:
                f.write(p + "\n")

        print("Dataset files are created successfully.")


if __name__ == '__main__':
    LGGSegDataset.createLGGSegDatasetFiles(r"F:\Projects\Datasets\Brain MRI segmentation (LGG Segmentation Dataset) by Mateusz Buda")

