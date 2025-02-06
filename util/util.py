import datetime
import time
import argparse
import numpy as np
from torchvision import transforms
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


def parseArgs(args=None, appname=None):
    if isinstance(args, str):
        args = args.split()

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size',
                        help='Batch size to use for training',
                        default=16,
                        type=int,
                        )
    parser.add_argument('--num-workers',
                        help='Number of worker processes for background data loading',
                        default=8,
                        type=int,
                        )
    parser.add_argument('--epochs',
                        help='Number of epochs to train for',
                        default=1,
                        type=int,
                        )

    parser.add_argument('comment',
                        help="Comment suffix for Tensorboard run.",
                        nargs='?',
                        default="test",
                        )
    parser.add_argument('--tb-prefix',
                        default=appname,
                        help="Data prefix to use for Tensorboard run. Defaults to chapter.",
                        )

    parser.add_argument('--augmented',
                        help="Augment the training data.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augment-flip',
                        help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augment-offset',
                        help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augment-scale',
                        help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augment-rotate',
                        help="Augment the training data by randomly rotating the data around the head-foot axis.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augment-noise',
                        help="Augment the training data by randomly adding noise to the data.",
                        action='store_true',
                        default=False,
                        )

    return parser.parse_args(args)


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


def compute_mIoU(CM, ignore_index=None):
    np.seterr(divide="ignore", invalid="ignore")
    if ignore_index is not None:
        CM = np.delete(CM, ignore_index, axis=0)
        CM = np.delete(CM, ignore_index, axis=1)
    intersection = np.diag(CM)
    ground_truth_set = CM.sum(axis=1)
    predicted_set = CM.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    mIoU = np.mean(IoU)
    return mIoU


def importstr(module_str, from_=None):
    """
    >>> importstr('os')
    <module 'os' from '.../os.pyc'>
    >>> importstr('math', 'fabs')
    <built-in function fabs>
    """
    if from_ is None and ':' in module_str:
        module_str, from_ = module_str.rsplit(':')

    module = __import__(module_str)
    for sub_str in module_str.split('.')[1:]:
        module = getattr(module, sub_str)

    if from_:
        try:
            return getattr(module, from_)
        except:
            raise ImportError('{}.{}'.format(module_str, from_))
    return module


def enumerateWithEstimate(
        iter_obj,
        desc_str,
        start_ndx=0,
        print_ndx=4,
        backoff=None,
        iter_len=None,
):
    """
    In terms of behavior, `enumerateWithEstimate` is almost identical
    to the standard `enumerate` (the differences are things like how
    our function returns a generator, while `enumerate` returns a
    specialized `<enumerate object at 0x...>`).

    However, the side effects (logging, specifically) are what make the
    function interesting.

    :param iter_obj: `iter` is the iterable that will be passed into
        `enumerate`. Required.

    :param desc_str: This is a human-readable string that describes
        what the loop is doing. The value is arbitrary, but should be
        kept reasonably short. Things like `"epoch 4 training"` or
        `"deleting temp files"` or similar would all make sense.

    :param start_ndx: This parameter defines how many iterations of the
        loop should be skipped before timing actually starts. Skipping
        a few iterations can be useful if there are startup costs like
        caching that are only paid early on, resulting in a skewed
        average when those early iterations dominate the average time
        per iteration.

        NOTE: Using `start_ndx` to skip some iterations makes the time
        spent performing those iterations not be included in the
        displayed duration. Please account for this if you use the
        displayed duration for anything formal.

        This parameter defaults to `0`.

    :param print_ndx: determines which loop interation that the timing
        logging will start on. The intent is that we don't start
        logging until we've given the loop a few iterations to let the
        average time-per-iteration a chance to stablize a bit. We
        require that `print_ndx` not be less than `start_ndx` times
        `backoff`, since `start_ndx` greater than `0` implies that the
        early N iterations are unstable from a timing perspective.

        `print_ndx` defaults to `4`.

    :param backoff: This is used to how many iterations to skip before
        logging again. Frequent logging is less interesting later on,
        so by default we double the gap between logging messages each
        time after the first.

        `backoff` defaults to `2` unless iter_len is > 1000, in which
        case it defaults to `4`.

    :param iter_len: Since we need to know the number of items to
        estimate when the loop will finish, that can be provided by
        passing in a value for `iter_len`. If a value isn't provided,
        then it will be set by using the value of `len(iter)`.

    :return:
    """
    if iter_len is None:
        iter_len = len(iter_obj)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    log.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))
    start_ts = time.time()
    for (current_ndx, item) in enumerate(iter_obj):
        yield current_ndx, item
        if current_ndx == print_ndx:
            # ... <1>
            duration_sec = ((time.time() - start_ts)
                            / (current_ndx - start_ndx + 1)
                            * (iter_len - start_ndx)
                            )

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            log.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_ndx,
                iter_len,
                str(done_dt).rsplit('.', 1)[0],
                str(done_td).rsplit('.', 1)[0],
            ))

            print_ndx *= backoff

        if current_ndx + 1 == start_ndx:
            start_ts = time.time()

    log.warning("{} ----/{}, done at {}".format(
        desc_str,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))
