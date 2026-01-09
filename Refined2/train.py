import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import RecordingSequenceDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np

class SequenceModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, num_labels, max_len=300):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, x, mask=None):
        x = self.input_proj(x) + self.pos_emb[:, : x.size(1), :]
        x = x.transpose(0, 1)
        src_key_padding_mask = ~mask if mask is not None else None
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        out = out.transpose(0, 1)
        logits = self.classifier(out)
        return logits


def train(processed_root: Path, epochs=8, batch_size=4, device='cuda'):
    ds = RecordingSequenceDataset(processed_root)
    labels_map = ds.label_to_id
    num_labels = len(labels_map)
    idxs = list(range(len(ds)))
    
    # Only split if we have enough samples
    if len(idxs) >= 2:
        tr_idx, val_idx = train_test_split(idxs, test_size=0.15, random_state=42)
        tr_ds = torch.utils.data.Subset(ds, tr_idx)
        val_ds = torch.utils.data.Subset(ds, val_idx)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: tuple(zip(*b)))
        use_validation = True
        print(f"Dataset split: {len(tr_idx)} train, {len(val_idx)} validation samples")
    else:
        # Use entire dataset for training (no validation split)
        tr_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))
        val_loader = None
        use_validation = False
        print(f"Warning: Only {len(ds)} sample(s) in dataset. Training without validation split.")

    sample_feat, _, _, _ = ds[0]
    input_dim = sample_feat.shape[1]
    model = SequenceModel(input_dim=input_dim, d_model=512, num_heads=8, num_layers=3, num_labels=num_labels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(1, epochs+1):
        model.train()
        tot = 0.0
        for batch in tqdm(tr_loader, desc=f"Train E{epoch}"):
            feats_batch, labels_batch, masks_batch, metas_batch = batch
            feats = torch.stack(feats_batch).to(device)
            labels = torch.stack(labels_batch).to(device)
            masks = torch.stack(masks_batch).to(device)
            # metas_batch is available but not used in training loss
            logits = model(feats, mask=masks)
            B, L, C = logits.shape
            loss = criterion(logits.view(B*L, C), labels.view(B*L))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"Epoch {epoch} train loss {tot/len(tr_loader):.4f}")
        
        # validation
        if use_validation:
            model.eval()
            vtot = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Val E{epoch}"):
                    feats_batch, labels_batch, masks_batch, metas_batch = batch
                    feats = torch.stack(feats_batch).to(device)
                    labels = torch.stack(labels_batch).to(device)
                    masks = torch.stack(masks_batch).to(device)
                    logits = model(feats, mask=masks)
                    B, L, C = logits.shape
                    loss = criterion(logits.view(B*L, C), labels.view(B*L))
                    vtot += loss.item()
                    
                    # Collect predictions and labels for metrics (excluding padding)
                    preds = logits.argmax(dim=-1)  # B x L
                    for b in range(B):
                        mask_b = masks_batch[b].cpu().numpy()
                        pred_b = preds[b].cpu().numpy()[mask_b]
                        label_b = labels_batch[b].cpu().numpy()[mask_b]
                        # Filter out ignore_index (-100)
                        valid_mask = label_b != -100
                        all_preds.extend(pred_b[valid_mask].tolist())
                        all_labels.extend(label_b[valid_mask].tolist())
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted', zero_division=0
            )
            accuracy = accuracy_score(all_labels, all_preds)
            
            print(f"Epoch {epoch} val loss {vtot/len(val_loader):.4f}")
            print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        torch.save({'model': model.state_dict(), 'labels_map': labels_map}, processed_root / f'model_epoch_{epoch}.pth')
    torch.save({'model': model.state_dict(), 'labels_map': labels_map}, processed_root / 'model_final.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_root', required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train(Path(args.processed_root), epochs=args.epochs, batch_size=args.batch_size, device=args.device)
