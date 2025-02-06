from util.util import importstr
from util.logconf import logging
log = logging.getLogger(__file__.rsplit('\\', 1)[-1])
log.setLevel(logging.DEBUG)

training_epochs = 20
experiment_epochs = 10
final_epochs = 50
seg_epochs = 10


def run(app, args):
    log.info("Running: {}({!r}).testRun()".format(app, args))

    app_cls = importstr(*app.rsplit('.', 1))  # <2>
    app_cls(args).testRun()

    log.warning("test")
    log.debug("test")
    log.info("Finished: {}({!r}).testRun()".format(app, args))

from dataset.example_dataset import ExampleDataset

def testExampleDataset():
    dataset = ExampleDataset(data_type='trn')
    print(len(dataset))
    img, lab = dataset[0]
    print(img.shape)
    print(lab.shape)
    print(img.dtype)


if __name__ == '__main__':
    pass
    # run('app.example_app.ExampleApp', f'--epochs={final_epochs} --batch-size=64 --augmented')
    testExampleDataset()
