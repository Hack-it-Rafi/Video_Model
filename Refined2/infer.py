from pathlib import Path
import numpy as np
import torch
from train import SequenceModel
from feature_extractor import ChunkFeatureExtractor
import argparse
import requests
from ollama_client import ollama_summarize


def load_model(processed_root: Path, device='cuda'):
    ck = processed_root / 'model_final.pth'
    if not ck.exists():
        pths = sorted(processed_root.glob('model_epoch_*.pth'))
        if not pths:
            raise FileNotFoundError('no model checkpoint')
        ck = pths[-1]
    ckdata = torch.load(ck, map_location='cpu')
    labels_map = ckdata.get('labels_map')
    id_to_label = {v:k for k,v in labels_map.items()}
    
    # Find recording directories (not .pth files) to get feature dimensions
    recording_dirs = [d for d in processed_root.iterdir() if d.is_dir() and (d / 'features.npy').exists()]
    if not recording_dirs:
        raise FileNotFoundError('No processed recordings found to determine input dimensions')
    sample_feat = np.load(recording_dirs[0] / 'features.npy')
    input_dim = sample_feat.shape[1]
    
    model = SequenceModel(input_dim=input_dim, d_model=512, num_heads=8, num_layers=3, num_labels=len(labels_map))
    model.load_state_dict(ckdata['model'])
    model = model.eval().to(device)
    return model, id_to_label


def infer_on_recording(recording_dir: Path, processed_root: Path, device='cuda'):
    out_dir = processed_root / recording_dir.name
    if not out_dir.exists() or not (out_dir / 'features.npy').exists():
        # extract features for this recording only
        from preprocess import parse_annotation_line
        extractor = ChunkFeatureExtractor(device=device, target_frames=50, image_size=112)
        ann = [l.strip() for l in open(recording_dir / 'annotations.csv','r',encoding='utf-8').read().splitlines() if l.strip()]
        feats = []
        labels = []
        metas = []
        for ln in ann:
            fname, lbl, meta = parse_annotation_line(ln)
            fpath = recording_dir / fname
            if not fpath.exists():
                fpath = recording_dir / 'chunks' / fname
            if not fpath.exists():
                continue
            v = extractor.extract(fpath)
            feats.append(v)
            labels.append(lbl)
            metas.append(meta)
        if len(feats) == 0:
            raise RuntimeError('no features extracted')
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / 'features.npy', np.stack(feats, axis=0))
        np.save(out_dir / 'labels.npy', np.array(labels, dtype=object))
        np.save(out_dir / 'meta.npy', np.array(metas, dtype=object))

    model, id_to_label = load_model(processed_root, device=device)
    feats = np.load(out_dir / 'features.npy')
    
    # Load metadata if available
    meta_path = out_dir / 'meta.npy'
    if meta_path.exists():
        metas = np.load(meta_path, allow_pickle=True)
    else:
        metas = np.array([''] * feats.shape[0], dtype=object)
    
    L = feats.shape[0]
    pad_to = 300
    if L < pad_to:
        pad = pad_to - L
        feats_p = np.pad(feats, ((0,pad),(0,0)), mode='constant')
        mask = np.concatenate([np.ones(L, dtype=bool), np.zeros(pad, dtype=bool)])
    else:
        feats_p = feats[:pad_to]
        mask = np.ones(pad_to, dtype=bool)
    x = torch.from_numpy(feats_p).unsqueeze(0).float().to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x, mask=mask_t)
        probs = torch.softmax(logits, dim=-1)[0]
        pred = probs.argmax(dim=-1).cpu().numpy()[:L]
        conf = probs.max(dim=-1).values.cpu().numpy()[:L]
    
    # collapse with metadata
    steps = []
    current = pred[0]
    confs = [conf[0]]
    current_metas = [metas[0]]
    for i in range(1, L):
        if pred[i] == current:
            confs.append(conf[i])
            current_metas.append(metas[i])
        else:
            # Join unique metadata values for this step
            unique_metas = list(dict.fromkeys([m for m in current_metas if m]))
            meta_str = ', '.join(unique_metas) if unique_metas else ''
            steps.append((id_to_label[int(current)], float(np.mean(confs)), meta_str))
            current = pred[i]
            confs = [conf[i]]
            current_metas = [metas[i]]
    # Final step
    unique_metas = list(dict.fromkeys([m for m in current_metas if m]))
    meta_str = ', '.join(unique_metas) if unique_metas else ''
    steps.append((id_to_label[int(current)], float(np.mean(confs)), meta_str))
    
    print('\n'.join([f"Step {i+1}: {s}{' [' + m + ']' if m else ''} (conf {c:.2f})" for i,(s,c,m) in enumerate(steps)]))

    
    try:
        # Convert steps to format expected by ollama_summarize (label, conf, meta)
        summary = ollama_summarize(steps, model_name='llama3.2')
        print('\n--- Ollama summary ---\n', summary)
    except Exception as e:
        print('Ollama summarization failed:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recording', required=True)
    parser.add_argument('--processed_root', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    infer_on_recording(Path(args.recording), Path(args.processed_root), device=args.device)
