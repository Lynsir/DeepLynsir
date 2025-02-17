import hashlib
import shutil
import sys
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from early_stopping_pytorch import EarlyStopping

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from util import util
from dataset.example_dataset import ExampleDataset
from paper.unet import UNet

log = util.logging.getLogger(__name__)
log.setLevel(util.logging.DEBUG)

EPSILON = 1e-6


class ExampleApp:
    def __init__(self, args=None, ):
        if not args:
            args = sys.argv[1:]

        self.args = util.parseArgs(args, self.__class__.__name__)

        self.time_str = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.trn_writer = None
        self.val_writer = None
        self.early_stopping = EarlyStopping(patience=7, path="", trace_func=log.info)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.batch_size = self.args.batch_size * torch.cuda.device_count() if self.use_cuda else self.args.batch_size

        self.model = self.initModel()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.initOptimizer()

        self.validation_cadence = 5
        self.totalTrnSamples_count = 0

        self.cm = np.empty((2, 2))
        self.loss_sum = util.AverageMeter()

    def initModel(self):
        model = UNet(
            in_channels=3,
            n_classes=2,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        model = nn.Sequential(model, nn.Sigmoid())

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            model = model.to(self.device)

        return model

    def initCriterion(self, cri_str="diceLoss"):
        if cri_str.lower() == "diceloss":
            from paper.loss import diceLoss
            criterion = diceLoss
        else:
            criterion = getattr(nn, cri_str)().to(self.device)

        return criterion

    def initOptimizer(self, lr=1E-3):
        return Adam(self.model.parameters(), lr=lr)
        # return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def initDataLoader(self, data_type="trn"):
        ds = ExampleDataset(data_type=data_type)
        shuffle = True if data_type == 'trn' else False

        return DataLoader(ds,
                          batch_size=self.batch_size,
                          num_workers=self.args.num_workers,
                          pin_memory=self.use_cuda,
                          shuffle=shuffle, )

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join(os.path.dirname(__file__), "..", 'runs', self.args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(log_dir=log_dir + '_trn_seg_' + self.args.comment)
            self.val_writer = SummaryWriter(log_dir=log_dir + '_val_seg_' + self.args.comment)

    def initCM(self):
        self.cm.fill(0)

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()

        self.initCM()
        self.loss_sum.reset()

        for batch_ndx, batch_tup in tqdm(enumerate(train_dl), ncols=100, total=len(train_dl),
                                         desc=f"Training   Epoch {epoch_ndx}", unit='batch'):
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(batch_tup)
            loss_var.backward()
            self.optimizer.step()

        self.logHistory(epoch_ndx, 'trn')

    def doValidation(self, epoch_ndx, val_dl):
        self.model.eval()

        with torch.no_grad():
            self.initCM()
            self.loss_sum.reset()

            for batch_ndx, batch_tup in tqdm(enumerate(val_dl), total=len(val_dl), ncols=100,
                                             desc=f"Validation Epoch {epoch_ndx}", unit='batch'):
                loss_g = self.computeBatchLoss(batch_tup)

        return self.logHistory(epoch_ndx, 'val'), loss_g

    def computeBatchLoss(self, batch_tup):
        input_t, label_t = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        prediction_g = self.model(input_g)
        loss_g = self.criterion(prediction_g, label_g.long())

        with torch.no_grad():
            self.loss_sum.update(loss_g.cpu().detach().numpy(), len(input_g))
            pred_g = torch.argmax(F.softmax(prediction_g, dim=1), dim=1)
            self.cm += confusion_matrix(label_g.flatten().cpu().numpy(), pred_g.flatten().cpu().numpy())

        return loss_g

    def computeMetrics(self):
        #   F   T
        # N TN  FP
        # P FN  TP
        tp = self.cm[1, 1]
        tn = self.cm[0, 0]
        fn = self.cm[1, 0]
        fp = self.cm[0, 1]

        metrics_dict = {'loss/all': self.loss_sum.mean(),
                        'loss/dsc_score': 2 * tp / (2 * tp + fp + fn + EPSILON),
                        'loss/iou_score': tp / (tp + fp + fn + EPSILON),
                        'percent_all/acc': (tp + tn) / (tp + tn + fp + fn + EPSILON),
                        'percent_all/tp': tp / (tp + fn + EPSILON),
                        'percent_all/tn': tn / (tn + fp + EPSILON),
                        }

        precision = metrics_dict['pr/precision'] = tp / (tp + fp + EPSILON)
        recall = metrics_dict['pr/recall'] = tp / (tp + fn + EPSILON)
        f1_score = metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall+ EPSILON)

        score = metrics_dict['loss/dsc_score']

        return metrics_dict, score

    def logHistory(self, epoch_ndx, mode_str):
        metrics_dict, score = self.computeMetrics()

        util.showMetrics(epoch_ndx, mode_str, metrics_dict)

        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')

        prefix_str = self.__class__.__name__ + '_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, epoch_ndx)

        writer.flush()

        return score

    def logImages(self, epoch_ndx, mode_str, dl):
        self.model.eval()
        dtset = dl.dataset
        for ndx in range(3):
            img_t, lab_t = dtset[ndx]
            pred_g = self.model(img_t.unsqueeze(0).to(self.device))

            pred_a = torch.argmax(F.softmax(pred_g, dim=1), dim=1).squeeze(0).detach().cpu().numpy()
            lab_a = lab_t.numpy()

            img_a = img_t.numpy().copy().transpose(1, 2, 0)
            img_a[:, :, 0] += pred_a & (1 - lab_a)
            img_a[:, :, 1] += pred_a & lab_a
            img_a[:, :, 2] += (1 - pred_a) & lab_a

            img_a *= 0.5
            img_a.clip(0, 1, img_a)

            writer = getattr(self, mode_str + '_writer')
            writer.add_image(f'{mode_str}/pred_{ndx}: {dtset.datalines[ndx]}', img_a, epoch_ndx, dataformats='HWC')

            if epoch_ndx == 1:
                img_a = img_t.numpy().copy().transpose(1, 2, 0)
                img_a[:, :, 1] += lab_a  # Green
                img_a *= 0.5
                img_a.clip(0, 1, img_a)
                writer.add_image(f'{mode_str}/label_{ndx}: {dtset.datalines[ndx]}', img_a, epoch_ndx, dataformats='HWC')

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'save',
            'models',
            self.args.tb_prefix,
            self.time_str,
            '{}_{}_{}.{}.state'.format(type_str, self.time_str, self.args.comment, self.totalTrnSamples_count)
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
            'totalTrainingSamples_count': self.totalTrnSamples_count,
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

        train_dl = self.initDataLoader('trn')
        val_dl = self.initDataLoader('val')

        best_score = 0.0
        for epoch_ndx in range(1, self.args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.args.epochs,
                len(train_dl),
                len(val_dl),
                self.args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            self.doTraining(epoch_ndx, train_dl)

            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                # if validation is wanted
                score, loss_g = self.doValidation(epoch_ndx, val_dl)
                best_score = max(score, best_score)

                # self.saveModel('seg', epoch_ndx, score == best_score)

                self.logImages(epoch_ndx, 'trn', train_dl)
                self.logImages(epoch_ndx, 'val', val_dl)

                self.early_stopping(loss_g, self.model)
                if self.early_stopping.early_stop:
                    log.info("Early stopping in Epoch {} of {}".format(epoch_ndx, self.args.epochs))
                    break

        self.trn_writer.close()
        self.val_writer.close()

        log.info("Training completed. Good luck!")


if __name__ == "__main__":
    #  python -m app.example_app --epochs=1 --batch-size=8 test
    ExampleApp().run()
