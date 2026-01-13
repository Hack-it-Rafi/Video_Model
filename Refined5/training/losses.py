import torch.nn as nn

class MultiHeadLoss(nn.Module):
    def __init__(self, w_action, w_app):
        super().__init__()
        self.la = nn.CrossEntropyLoss(weight=w_action)
        self.lp = nn.CrossEntropyLoss(weight=w_app)
    def forward(self, pa, pp, ta, tp):
        return self.la(pa, ta) + self.lp(pp, tp)