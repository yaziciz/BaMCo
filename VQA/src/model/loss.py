import torch
import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match\n' + str(predict.shape) + '\n' + str(target.shape)
        target_ = target.clone()
        target_[target == -1] = 0

        ce_loss = self.criterion(predict, target_.float())

        return ce_loss
