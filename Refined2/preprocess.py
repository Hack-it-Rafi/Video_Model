from pathlib import Path
import numpy as np
import pandas as pd
from feature_extractor import ChunkFeatureExtractor
import argparse


#annotations.csv   # lines: chunk_000.mp4 | app:switch | notepad


def parse_annotation_line(line: str):
    # '|' or ','
    parts = [p.strip() for p in line.split(',')]
    if len(parts) < 2:
        parts = [p.strip() for p in line.split(',')]
    
    fname = parts[0]
    label = parts[1]
    meta = parts[2] if len(parts) > 2 else ''
    return fname, label, meta


def preprocess_dataset(dataset_root: Path, processed_root: Path, device='cuda'):
    dataset_root = Path(dataset_root)
    processed_root = Path(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    extractor = ChunkFeatureExtractor(device=device, target_frames=50, image_size=112)

    for rec in sorted(dataset_root.iterdir()):
        if not rec.is_dir():
            continue
        ann_path = rec / 'annotations.csv'
        if not ann_path.exists():
            print(f"Skip {rec.name}: annotations.csv missing")
            continue
        df_lines = [l for l in open(ann_path, 'r', encoding='utf-8').read().strip().splitlines() if l.strip()]
        chunk_files = []
        labels = []
        metas = []
        for ln in df_lines:
            fname, lbl, meta = parse_annotation_line(ln)
            fpath = rec / fname
            if not fpath.exists():
                fpath = rec / 'chunks' / fname
                if not fpath.exists():
                    print(f"Warning: chunk file not found for {fname} in {rec}")
                    continue
            chunk_files.append(fpath)
            labels.append(lbl)
            metas.append(meta)

        feats = []
        for p in chunk_files:
            try:
                fv = extractor.extract(p)
                feats.append(fv)
            except Exception as e:
                print(f"Failed extract {p}: {e}")
        if len(feats) == 0:
            print(f"No features for {rec.name}, skipping save")
            continue
        feats = np.stack(feats, axis=0)
        labels = np.array(labels, dtype=object)
        metas = np.array(metas, dtype=object)
        out = processed_root / rec.name
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / 'features.npy', feats)
        np.save(out / 'labels.npy', labels)
        np.save(out / 'meta.npy', metas)
        print(f"Saved {out} features {feats.shape} labels {labels.shape} meta {metas.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--processed_root', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    preprocess_dataset(Path(args.dataset_root), Path(args.processed_root), device=args.device)
