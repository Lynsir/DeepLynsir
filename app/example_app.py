import hashlib
import shutil
import sys
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from util import util
from paper.lossfunc import diceLoss
from dataset.example_dataset import ExampleDataset
from paper.unet import UNet

log = util.logging.getLogger(__name__)
log.setLevel(util.logging.DEBUG)

# Used for computeClassificationLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_LOSS_NDX = 1
# METRICS_FN_LOSS_NDX = 2
# METRICS_ALL_LOSS_NDX = 3

# METRICS_PTP_NDX = 4
# METRICS_PFN_NDX = 5
# METRICS_MFP_NDX = 6

METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9
METRICS_TN_NDX = 10

METRICS_DSC_NDX = 11

METRICS_SIZE = 12


class ExampleApp:
    def __init__(self, args=None,):
        if not args:
            args = sys.argv[1:]

        self.args = util.parseArgs(args, self.__class__.__name__)

        self.time_str = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.trn_writer = None
        self.val_writer = None

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

        self.validation_cadence = 5
        self.totalTrainingSamples_count = 0

    def initModel(self):
        # 使用标准U-net结构，五层深度，每层64个通道
        model = UNet(
            in_channels=3,
            n_classes=1,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)

        return nn.Sequential(model, nn.Sigmoid())


    def initOptimizer(self):
        return Adam(self.model.parameters())
        # return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def initTrainDl(self):
        trn_ds = ExampleDataset(data_type='trn')
        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            trn_ds,
            batch_size=batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.use_cuda,
            shuffle=True,
        )

        return train_dl

    def initValDl(self):
        val_ds = ExampleDataset(data_type='val')
        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.use_cuda,
            shuffle=False,
        )

        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join(os.path.dirname(__file__), "..", 'runs', self.args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(log_dir=log_dir + '_trn_seg_' + self.args.comment)
            self.val_writer = SummaryWriter(log_dir=log_dir + '_val_seg_' + self.args.comment)

    def doTraining(self, epoch_ndx, train_dl):
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        self.model.train()

        batch_iter = util.enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()

            self.optimizer.step()

        self.totalTrainingSamples_count += trnMetrics_g.size(1)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            self.model.eval()

            batch_iter = util.enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g,
                         classificationThreshold=0.5):
        input_t, label_t = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        prediction_g = self.model(input_g)

        diceLoss_g = diceLoss(prediction_g, label_g)
        fnLoss_g = diceLoss(prediction_g * label_g, label_g)

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        with torch.no_grad():
            # fix:原来使用的是prediction_g[:, 0:1]切片产生的形状是[B,1,H,W]，导致后续的计算会出错，修正后的形状是[B,H,W]
            predictionBool_g = (prediction_g[:, 0] > classificationThreshold).to(torch.float32)

            # 对最后两个维度求和, 因为是二维图像
            tp = (predictionBool_g * label_g).sum(dim=[-1, -2])
            tn = ((1 - predictionBool_g) * (~label_g)).sum(dim=[-1, -2])
            fn = ((1 - predictionBool_g) * label_g).sum(dim=[-1, -2])
            fp = (predictionBool_g * (~label_g)).sum(dim=[-1, -2])

            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_g
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_TN_NDX, start_ndx:end_ndx] = tn
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp


            # TODO:计算骰子系数并记录
            # dice = 2TP/(FN+FP+2TP) = 2真阳性/(实际阳性(TP+FN)+预测阳性(TP+FP))

        # fnLoss_g是真阳性与实际阳性的骰子系数，如果他的值越小说明真阳越多
        # 在损失中加上8倍的这个值以惩罚阴性样本带来的影响，因为阴性的样本数量远多于阳性
        # 预测的阴性越多，说明真阳越少，惩罚就越大，损失就越大
        # 可以理解为阳性比阴性重要8倍，不管怎么优化阴性的正确率，损失始终不会降低太多
        # return diceLoss_g.mean() + fnLoss_g.mean() * 8
        return diceLoss_g.mean() + fnLoss_g.mean() * 0
        # return diceLoss_g.mean()

    def logImages(self, epoch_ndx, mode_str, dl):
        self.model.eval()
        dtset = dl.dataset
        for ndx in range(3):
            img_t, lab_t = dtset[ndx]
            img_g = img_t.to(self.device)
            # 构造输出形状为[1,1,H,W]，并转换为H,W
            pred_g = self.model(img_g.unsqueeze(0))[0,0]

            pred_a = pred_g.detach().cpu().numpy()>0.5
            lab_a = lab_t.numpy()>0.5

            # 监测数据使用
            # P = lab_a.sum()
            # N = (~lab_a).sum()
            # TP = (pred_a & lab_a).sum()
            # TN = (~pred_a & ~lab_a).sum()
            # FP = (pred_a & ~lab_a).sum()
            # FN = (~pred_a & lab_a).sum()
            #
            # log.info(f"E{epoch_ndx} {mode_str} "
            #          f"TP:{TP}/{P}={TP / P :.2%} "
            #          f"TN:{TN}/{N}={TN / N:.2%} "
            #          f"ACC:{(TP + TN) / (P + N):.2%} "
            #          f"DSC:{(2 * TP) / (2*TP + FN + FP):.2%}")

            # 转换成H,W,C
            img_a = img_t.numpy().copy().transpose(1,2,0)
            # 假阳性修改为红色，0通道表示R
            img_a[:, :, 0] += pred_a & (1 - lab_a)
            # 真阳性修改为绿色，1通道表示G
            img_a[:, :, 1] += pred_a & lab_a
            # 假阴性修改为蓝色，2通道表示B
            img_a[:, :, 2] += (1 - pred_a) & lab_a

            img_a *= 0.5
            img_a.clip(0, 1, img_a)

            writer = getattr(self, mode_str + '_writer')
            writer.add_image(
                f'{mode_str}/prediction_{ndx}:    {dtset.datalines[ndx]}',
                img_a,
                epoch_ndx,
                dataformats='HWC',
            )

            if epoch_ndx == 1:
                # fix: 修复标签图像变红问题
                # 原因：由于img_t底层内存共享数据，上面修改img_a会影响img_t，导致下面的img_a获取的是修改后的数据
                # 解决方法：调用numpy的copy方法，创建一个新的内存空间，避免共享数据
                img_a = img_t.numpy().copy().transpose(1, 2, 0)
                img_a[:, :, 1] += lab_a  # Green
                img_a *= 0.5
                img_a.clip(0, 1, img_a)
                writer.add_image(
                    f'{mode_str}/label_{ndx}:    {dtset.datalines[ndx]}',
                    img_a,
                    epoch_ndx,
                    dataformats='HWC',
                )


    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        allPos_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]
        allNeg_count = sum_a[METRICS_TN_NDX] + sum_a[METRICS_FP_NDX]
        all_count = allPos_count + allNeg_count
        true_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_TN_NDX]

        metrics_dict = {'loss/all': metrics_a[METRICS_LOSS_NDX].mean(),
                        'percent_all/acc': true_count / (all_count or 1) * 100,
                        'percent_all/tp': sum_a[METRICS_TP_NDX] / (allPos_count or 1) * 100,
                        'percent_all/tn': sum_a[METRICS_TN_NDX] / (allNeg_count or 1) * 100,
                        }

        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
                                                   / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall = metrics_dict['pr/recall'] = sum_a[METRICS_TP_NDX] \
                                             / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
                                      / ((precision + recall) or 1)

        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{pr/precision:.4f} precision, "
                  + "{pr/recall:.4f} recall, "
                  + "{pr/f1_score:.4f} f1"
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/acc:-5.1f}% ACC, {percent_all/tp:-5.1f}% TP, {percent_all/tn:-5.1f}% TN"
                  ).format(
            epoch_ndx,
            mode_str + '_all',
            **metrics_dict,
        ))

        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')

        # prefix_str = 'seg_'
        prefix_str = self.__class__.__name__ + '_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.totalTrainingSamples_count)

        writer.flush()

        # TODO: 这里使用召回率作为得分， 后续可以修改为其他指标
        score = metrics_dict['pr/recall']

        return score

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'save',
            'models',
            self.args.tb_prefix,
            self.time_str,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.args.comment,
                self.totalTrainingSamples_count,
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)

        log.info("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                'save', 'models',
                self.args.tb_prefix,
                self.time_str,
                f'{type_str}_{self.time_str}_{self.args.comment}.best.state')
            shutil.copyfile(file_path, best_path)

            log.info("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

    def run(self):

        log.info("Starting {}\n\t\t{}".format(type(self).__name__, self.args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        best_score = 0.0
        self.validation_cadence = 5
        for epoch_ndx in range(1, self.args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.args.epochs,
                len(train_dl),
                len(val_dl),
                self.args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                # if validation is wanted
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)

                # self.saveModel('seg', epoch_ndx, score == best_score)

                self.logImages(epoch_ndx, 'trn', train_dl)
                self.logImages(epoch_ndx, 'val', val_dl)

        self.trn_writer.close()
        self.val_writer.close()


if __name__ == "__main__":
    ExampleApp("--epochs 1 --batch-size 3 test_from_pyfile").run()
