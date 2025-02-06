import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def transform(IsResize, Resize_size, IsTotensor, IsNormalize, Norm_mean, Norm_std, IsRandomGrayscale, IsColorJitter,
              brightness, contrast, hue, saturation, IsCentercrop, Centercrop_size, IsRandomCrop, RandomCrop_size,
              IsRandomResizedCrop, RandomResizedCrop_size, Grayscale_rate, IsRandomHorizontalFlip, HorizontalFlip_rate,
              IsRandomVerticalFlip, VerticalFlip_rate, IsRandomRotation, degrees):
    transform_list = []

    # -----------------------------------------------<旋转图像>-----------------------------------------------------------#
    if IsRandomRotation:
        transform_list.append(transforms.RandomRotation(degrees))
    if IsRandomHorizontalFlip:
        transform_list.append(transforms.RandomHorizontalFlip(HorizontalFlip_rate))
    if IsRandomVerticalFlip:
        transform_list.append(transforms.RandomHorizontalFlip(VerticalFlip_rate))

    # -----------------------------------------------<图像颜色>-----------------------------------------------------------#
    if IsColorJitter:
        transform_list.append(transforms.ColorJitter(brightness, contrast, saturation, hue))
    if IsRandomGrayscale:
        transform_list.append(transforms.RandomGrayscale(Grayscale_rate))

    # ---------------------------------------------<缩放或者裁剪>----------------------------------------------------------#
    if IsResize:
        transform_list.append(transforms.Resize(Resize_size))
    if IsCentercrop:
        transform_list.append(transforms.CenterCrop(Centercrop_size))
    if IsRandomCrop:
        transform_list.append(transforms.RandomCrop(RandomCrop_size))
    if IsRandomResizedCrop:
        transform_list.append(transforms.RandomResizedCrop(RandomResizedCrop_size))

    # ---------------------------------------------<tensor化和归一化>------------------------------------------------------#
    if IsTotensor:
        transform_list.append(transforms.ToTensor())
    if IsNormalize:
        transform_list.append(transforms.Normalize(Norm_mean, Norm_std))

    # 您可以更改数据增强的顺序，但是数据增强的顺序可能会影响最终数据的质量，因此除非您十分明白您在做什么,否则,请保持默认顺序
    # transforms_order=[Resize_transform,Rotation,Color,Tensor,Normalize]
    return transforms.Compose(transform_list)


def get_transform(size=(200, 200), mean=(0, 0, 0), std=(1, 1, 1), IsResize=False, IsCentercrop=False,
                  IsRandomCrop=False, IsRandomResizedCrop=False, IsTotensor=False, IsNormalize=False,
                  IsRandomGrayscale=False, IsColorJitter=False, IsRandomVerticalFlip=False,
                  IsRandomHorizontalFlip=False, IsRandomRotation=False):
    diy_transform = transform(
        IsResize=IsResize,  # 是否缩放图像
        Resize_size=size,  # 缩放后的图像大小 如（512,512）->（256,192）
        IsCentercrop=IsCentercrop,  # 是否进行中心裁剪
        Centercrop_size=size,  # 中心裁剪后的图像大小
        IsRandomCrop=IsRandomCrop,  # 是否进行随机裁剪
        RandomCrop_size=size,  # 随机裁剪后的图像大小
        IsRandomResizedCrop=IsRandomResizedCrop,  # 是否随机区域进行裁剪
        RandomResizedCrop_size=size,  # 随机裁剪后的图像大小
        IsTotensor=IsTotensor,  # 是否将PIL和numpy格式的图片的数值范围从[0,255]->[0,1],且将图像形状从[H,W,C]->[C,H,W]
        IsNormalize=IsNormalize,  # 是否对图像进行归一化操作,即使用图像的均值和方差将图像的数值范围从[0,1]->[-1,1]
        Norm_mean=mean,  # 图像的均值，用于图像归一化，建议使用自己通过计算得到的图像的均值
        Norm_std=std,  # 图像的方差，用于图像归一化，建议使用自己通过计算得到的图像的方差
        IsRandomGrayscale=IsRandomGrayscale,  # 是否随机将彩色图像转化为灰度图像
        Grayscale_rate=0.5,  # 每张图像变成灰度图像的概率，设置为1的话等同于transforms.Grayscale()
        IsColorJitter=IsColorJitter,  # 是否随机改变图像的亮度、对比度、色调和饱和度
        brightness=0.5,  # 每个图像被随机改变亮度的概率
        contrast=0.5,  # 每个图像被随机改变对比度的概率
        hue=0.5,  # 每个图像被随机改变色调的概率
        saturation=0.5,  # 每个图像被随机改变饱和度的概率
        IsRandomVerticalFlip=IsRandomVerticalFlip,  # 是否垂直翻转图像
        VerticalFlip_rate=0.5,  # 每个图像被垂直翻转图像的概率
        IsRandomHorizontalFlip=IsRandomHorizontalFlip,  # 是否水平翻转图像
        HorizontalFlip_rate=0.5,  # 每个图像被水平翻转图像的概率
        IsRandomRotation=IsRandomRotation,  # 是是随机旋转图像
        degrees=10,  # 每个图像被旋转角度的范围 如degrees=10 则图像将随机旋转一个(-10,10)之间的角度
    )
    return diy_transform


class SegmentationAugmentation(nn.Module):
    def __init__(
            self, flip=None, offset=None, scale=None, rotate=None, noise=None
    ):
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g, label_g):
        transform_t = self._build2dTransformMatrix()
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        transform_t = transform_t.to(input_g.device, torch.float32)
        affine_t = F.affine_grid(transform_t[:,:2],
                input_g.size(), align_corners=False)

        augmented_input_g = F.grid_sample(input_g,
                affine_t, padding_mode='border',
                align_corners=False)
        augmented_label_g = F.grid_sample(label_g.to(torch.float32),
                affine_t, padding_mode='border',
                align_corners=False)

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise

            augmented_input_g += noise_t

        return augmented_input_g, augmented_label_g > 0.5

    def _build2dTransformMatrix(self):
        transform_t = torch.eye(3)

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i,i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)
                transform_t[2,i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = (random.random() * 2 - 1)
                transform_t[i,i] *= 1.0 + scale_float * random_float

        if self.rotate:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])

            transform_t @= rotation_t

        return transform_t
    
def getAugmentationSetting(args):
    augmentation_dict = {}

    if args.augmented or args.augment_flip:
        augmentation_dict['flip'] = True
    if args.augmented or args.augment_offset:
        augmentation_dict['offset'] = 0.03
    if args.augmented or args.augment_scale:
        augmentation_dict['scale'] = 0.2
    if args.augmented or args.augment_rotate:
        augmentation_dict['rotate'] = True
    if args.augmented or args.augment_noise:
        augmentation_dict['noise'] = 25.0

    return augmentation_dict