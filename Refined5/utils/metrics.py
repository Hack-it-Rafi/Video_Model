import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def head_metrics(pred, true):
    p = pred.argmax(1)
    acc = accuracy_score(true, p)
    pr,rc,f,_ = precision_recall_fscore_support(true,p,average='macro')
    return acc,pr,rc,f

def compute_metrics(pa,pp,ta,tp):
    return {
        "action": head_metrics(pa,ta),
        "app": head_metrics(pp,tp)
    }