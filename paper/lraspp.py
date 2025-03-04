from torch import nn
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


class LRASPP(nn.Module):
    def __init__(self, n_classes):
        super(LRASPP, self).__init__()
        self.model = lraspp_mobilenet_v3_large(weights=None, weights_backbon=None, num_classes=n_classes)

    def forward(self, x):
        return self.model(x)['out']
