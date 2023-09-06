import torch
from models.UNet_3Plus.UNet_3Plus import UNet_3Plus

class Model(torch.nn.Module):
    def __init__(self, in_channels, n_classes, filters):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.netG = UNet_3Plus(in_channels=in_channels, n_classes=n_classes, filters=filters).to(device)
            
    def forward(self, input):
        outputs = {}
        outputs['preds'] = self.netG(input['multis'])
        return outputs
