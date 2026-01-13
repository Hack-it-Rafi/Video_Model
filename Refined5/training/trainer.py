import torch
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, opt, loss_fn, device, mp):
        self.model, self.opt, self.loss_fn = model, opt, loss_fn
        self.device, self.mp = device, mp
        self.scaler = GradScaler() if mp else None

    def train_epoch(self, loader):
        self.model.train()
        total=0
        for v,a,p in loader:
            v,a,p = v.to(self.device), a.to(self.device), p.to(self.device)
            self.opt.zero_grad()
            if self.mp:
                with autocast(): pa,pp = self.model(v)
                loss = self.loss_fn(pa,pp,a,p)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                pa,pp = self.model(v); loss = self.loss_fn(pa,pp,a,p)
                loss.backward(); self.opt.step()
            total += loss.item()
        return total/len(loader)