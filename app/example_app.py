import hashlib
import shutil
import sys
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg.cython_lapack import dsycon
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from util import util
from util.augmentation import SegmentationAugmentation, getAugmentationSetting
from util.lossfunc import diceLoss
from dataset.example_dataset import ExampleDataset
from stdmodel.unet import UNet

log = util.logging.getLogger(__name__)
log.setLevel(util.logging.DEBUG)

# Used for computeClassificationLoss and logMetrics to index into metrics_t/metrics_a
# METRICS_LABEL_NDX = 0
METRICS_LOSS_NDX = 1
# METRICS_FN_LOSS_NDX = 2
# METRICS_ALL_LOSS_NDX = 3

# METRICS_PTP_NDX = 4
# METRICS_PFN_NDX = 5
# METRICS_MFP_NDX = 6
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9

METRICS_SIZE = 10


class ExampleApp:
    def __init__(self, args=None, is_test_run=True):
        if not args:
            args = sys.argv[1:]

        self.args = util.parseArgs(args, self.__class__.__name__)

        if not is_test_run:
            self.time_str = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
            self.trn_writer = None
            self.val_writer = None

            self.augmentation_dict = getAugmentationSetting(self.args)

            self.use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if self.use_cuda else "cpu")

            self.model, self.augmentation_model = self.initModel()
            self.optimizer = self.initOptimizer()

            self.validation_cadence = 5
            self.totalTrainingSamples_count = 0

    def initModel(self):
        model = UNet(
            in_channels=3,
            n_classes=1,
            depth=4,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                augmentation_model = nn.DataParallel(augmentation_model)
            model = model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)

        return model, augmentation_model


    def initOptimizer(self):
        return Adam(self.model.parameters())
        # return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def initTrainDl(self):
        trn_ds = ExampleDataset()
        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            trn_ds,
            batch_size=batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self):
        val_ds = ExampleDataset()
        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(log_dir=log_dir + '_trn_seg_' + self.args.comment)
            self.val_writer = SummaryWriter(log_dir=log_dir + '_val_seg_' + self.args.comment)

    def doTraining(self, epoch_ndx, train_dl):
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        self.model.train()
        train_dl.dataset.shuffleSamples()

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
        input_t, label_t, series_list, _slice_ndx_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        if self.model.training and self.augmentation_dict:
            input_g, label_g = self.augmentation_model(input_g, label_g)

        prediction_g = self.model(input_g)

        diceLoss_g = diceLoss(prediction_g, label_g)
        fnLoss_g = diceLoss(prediction_g * label_g, label_g)

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        with torch.no_grad():
            predictionBool_g = (prediction_g[:, 0:1]
                                > classificationThreshold).to(torch.float32)

            tp = (predictionBool_g * label_g).sum(dim=[1, 2, 3])
            fn = ((1 - predictionBool_g) * label_g).sum(dim=[1, 2, 3])
            fp = (predictionBool_g * (~label_g)).sum(dim=[1, 2, 3])

            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_g
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

        # fnLoss_g是真阳性与实际阳性的骰子系数，如果他的值越小说明真阳越多
        # 在损失中加上8倍的这个值以惩罚阴性样本带来的影响，因为阴性的样本数量远多于阳性
        # 预测的阴性越多，说明真阳越少，惩罚就越大，损失就越大
        # 可以理解为阳性比阴性重要8倍，不管怎么优化阴性的正确率，损失始终不会降低太多
        # return diceLoss_g.mean() + fnLoss_g.mean() * 8
        return diceLoss_g.mean()

    def logImages(self, epoch_ndx, mode_str, dl):
        self.model.eval()
        dtset = dl.dataset
        for ndx in range(3):
            img_t, lab_t = dtset[ndx]
            img_g = img_t.to(self.device)
            pred_g = self.model(img_g.unsqueeze(0))[0]

            pred_a = pred_g.detach().cpu().numpy()[0]>0.5
            lab_a = lab_t.numpy()[0]>0.5

            img_a = img_t.numpy().transpose(1,2,0)
            # 假阳性修改为红色，0通道表示R
            img_a[:, :, 0] += pred_a & (1 - lab_a)
            # 真阳性修改为绿色，1通道表示G
            img_a[:, :, 1] += pred_a & lab_a

            img_a *= 0.5
            img_a.clip(0, 1, img_a)

            writer = getattr(self, mode_str + '_writer')
            writer.add_image(
                f'{mode_str}/prediction_{ndx}:{dtset.datalines[ndx]}',
                img_a,
                self.totalTrainingSamples_count,
                dataformats='HWC',
            )

            if epoch_ndx == 1:
                img_a = img_t.numpy().transpose(1, 2, 0)
                img_a[:, :, 1] += lab_a  # Green
                img_a *= 0.5
                img_a.clip(0, 1, img_a)
                writer.add_image(
                    f'{mode_str}/label_{ndx}:{dtset.datalines[ndx]}',
                    img_a,
                    self.totalTrainingSamples_count,
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

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {'loss/all': metrics_a[METRICS_LOSS_NDX].mean(),
                        'percent_all/tp': sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100,
                        'percent_all/fn': sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100,
                        'percent_all/fp': sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100}

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
                  + "{pr/f1_score:.4f} f1 score"
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
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

        score = metrics_dict['pr/recall']

        return score

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'save',
            'models',
            self.args.tb_prefix,
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
                f'{type_str}_{self.time_str}_{self.args.comment}.best.state')
            shutil.copyfile(file_path, best_path)

            log.info("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

    def run(self):

        log.info("Starting {}, {}".format(type(self).__name__, self.args))

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

                self.saveModel('seg', epoch_ndx, score == best_score)

                self.logImages(epoch_ndx, 'trn', train_dl)
                self.logImages(epoch_ndx, 'val', val_dl)

        self.trn_writer.close()
        self.val_writer.close()


if __name__ == "__main__":
    ExampleApp().run()
