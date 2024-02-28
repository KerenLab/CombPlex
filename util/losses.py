import torch
import torch.nn as nn

class Loss(torch.nn.Module):
    # initializing the specific loss according to imaging platform  and model type
    def __init__(self, config):
        super().__init__()
        if config['model_type'] == 'Decompression masking network':
            self.binary_input = True
            self.loss = nn.BCELoss()
            self.loss_name = 'BCE'
        elif config['model_type'] == 'Value reconstruction network':
            self.binary_input = False
            if config['imaging_platform'] == 'CODEX':
                self.loss = nn.MSELoss()
                self.loss_name = 'MSE'
            elif config['imaging_platform'] == 'MIBI':
                self.loss = nn.PoissonNLLLoss(log_input=False, full=True)
                self.loss_name = 'PoissonNLL'
    
    def get_loss_name(self):
        return self.loss_name

    def forward(self, outputs, inputs):
        losses = {}
        preds = outputs['preds']

        # Binarize input in case of masking network
        if self.binary_input:
            inputs = (inputs['GT_singles'] > 0).float()
        else:
            inputs = inputs['GT_singles']

        losses[f'loss_{self.loss_name}'] = self.loss(preds, inputs)
        return losses