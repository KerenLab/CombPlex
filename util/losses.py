import torch
import torch.nn as nn

def BCE_loss(pred, label):
    bce_loss = nn.BCELoss()
    return bce_loss(pred, label)

class LossG(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, outputs, inputs):
        losses = {}
        loss_G = 0
        preds = outputs['preds']
        inputs = (inputs['GT_singles'] > 0).float()

        losses['loss_BCE'] = BCE_loss(preds, inputs)
        loss_G += losses['loss_BCE']

        losses['loss'] = loss_G
        return losses