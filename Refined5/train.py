import torch, yaml, os
from data.dataset import ScreenVideoDataset
from models.timesformer_multhead import TimeSformerMultiHead
from training.trainer import Trainer
from training.losses import MultiHeadLoss
from training.evaluate import evaluate
from utils.split import split_by_recording
from torch.utils.data import DataLoader

cfg=yaml.safe_load(open("configs/config.yaml"))
root="data"
folders=os.listdir(root)
train_f,val_f = split_by_recording(folders)

act_map={n:i for i,n in enumerate(cfg['labels']['actions'])}
app_map={n:i for i,n in enumerate(cfg['labels']['apps'])}

train_ds=ScreenVideoDataset(root,train_f,act_map,app_map)
val_ds=ScreenVideoDataset(root,val_f,act_map,app_map)

train_dl=DataLoader(train_ds,batch_size=cfg['project']['batch_size'],shuffle=True)
val_dl=DataLoader(val_ds,batch_size=cfg['project']['batch_size'])

model=TimeSformerMultiHead(len(act_map),len(app_map)).cuda()
opt=torch.optim.AdamW(model.parameters(),lr=cfg['project']['lr'])
w_action=torch.ones(len(act_map)).cuda()
w_app=torch.ones(len(app_map)).cuda()
loss_fn=MultiHeadLoss(w_action,w_app)
trainer=Trainer(model,opt,loss_fn,'cuda',cfg['project']['mixed_precision'])

for e in range(cfg['project']['epochs']):
    l=trainer.train_epoch(train_dl)
    m=evaluate(model,val_dl,'cuda')
    print(e,l,m)