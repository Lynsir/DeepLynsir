import datetime
import time
import argparse
import numpy as np
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
                        default=4,
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
                        default="testrun",
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

    return parser.parse_args(args)

# -------------------------------------------------------------------------------

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


# a class for calculating the average of the accuracy and the loss
# -------------------------------------------------------------------------------
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def mean(self):
        return self.sum / self.count


# -------------------------------------------------------------------------------

def showMetrics(epoch_ndx, mode_str, metrics_dict):
    log.info(("E{} {:5} "
              + "{loss/all:.4f} LOSS, "
              + "{metrics/mcc_score:.4f} MCC, "
              + "{metrics/dsc_score:.4f} DSC, "
              + "{metrics/iou_score:.4f} IOU"
              ).format(epoch_ndx, mode_str, **metrics_dict, ))
    log.info(("E{} {:5} "
              + "{pr/precision:.4f} PRC , "
              + "{pr/recall:.4f} RCL, "
              + "{pr/f1_score:.4f} F1"
              ).format(epoch_ndx, mode_str, **metrics_dict, ))
    log.info(("E{} {:5} "
              + "{percent_all/acc:-6.2%} ACC , "
                "{percent_all/tp:-6.2%} TP , "
                "{percent_all/tn:-6.2%} TN"
              ).format(epoch_ndx, mode_str, **metrics_dict, ))


# -------------------------------------------------------------------------------


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
