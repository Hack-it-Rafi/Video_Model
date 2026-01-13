import torch
from utils.metrics import compute_metrics

def evaluate(model, loader, device):
    model.eval(); A,P,TA,TP=[],[],[],[]
    with torch.no_grad():
        for v,a,p in loader:
            v=v.to(device); pa,pp = model(v)
            A.append(pa.cpu()); P.append(pp.cpu())
            TA.append(a); TP.append(p)
    return compute_metrics(torch.cat(A),torch.cat(P),torch.cat(TA),torch.cat(TP))