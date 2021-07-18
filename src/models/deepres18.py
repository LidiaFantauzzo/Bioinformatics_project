 
import torch.nn as nn
import torch.nn.functional as functional
from .deeplab import DeeplabV3
from .resnet import ResNet18

def DeepLabV3_ResNet18(args):

    body = ResNet18(args.in_channels)
    body.pretrain()

    head_channels = 256
    head = DeeplabV3(512, head_channels, 256, out_stride=16)

    model = CompleteNetwork(body, head, head_channels, args.num_classes)

    return model

class CompleteNetwork(nn.Module):

    def __init__(self, body, head, head_channels, classes):
        super(CompleteNetwork, self).__init__()
        self.body = body
        self.head = head
        self.cls = nn.Conv2d(head_channels, classes, 1)

        self.classes = classes

    def _network(self, x, ret_intermediate=False):

        x_b = self.body(x)
        x_o = self.head(x_b)

        if ret_intermediate:
            return x_b, x_o
        return x_o

    def freeze(self):
        for par in self.parameters():
            par.requires_grad = False

    def forward(self, x, ret_intermediate=False):
        out_size = x.shape[-2:]

        out = self._network(x, ret_intermediate)

        sem_logits = self.cls(out[1] if ret_intermediate else out)
        sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)

        if ret_intermediate:
            return sem_logits, {"body": out[0]}

        return sem_logits