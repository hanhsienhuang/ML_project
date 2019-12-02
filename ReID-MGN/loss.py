from opt import opt
import torch
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.nn.modules import loss
from utils.TripletLoss import TripletLoss
from utils.CenterLoss import CenterLoss
from utils.LSCrossEntropy import CrossEntropyLabelSmooth


class Loss(loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, outputs, labels):
        cross_entropy_loss = CrossEntropyLoss() if opt.label_smooth_ce else CrossEntropyLabelSmooth()
        triplet_loss = TripletLoss(margin=1.2)
        center_loss = CenterLoss()
        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[1:4]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[4:]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        loss_sum = Triplet_Loss + 2 * CrossEntropy_Loss

        if opt.use_centerloss:
            center_loss = [center_loss(output, labels) for output in outputs[1:4]]
            center_loss = sum(center_loss) / len(center_loss)
            loss_sum += 0.0005 * center_loss

        print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy()),
              end=' ')
        return loss_sum
