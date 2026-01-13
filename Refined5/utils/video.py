# shared video utilities (resize, normalize if needed)
import cv2
import torch
import numpy as np

NUM_FRAMES = 50
IMG_SIZE = 112

def read_video_fixed(path, num_frames=NUM_FRAMES, size=IMG_SIZE):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (size, size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Empty or broken video: {path}")

    frames = np.stack(frames)  # T, H, W, C

    # Enforce exactly num_frames
    if len(frames) > num_frames:
        idx = np.linspace(0, len(frames)-1, num_frames).astype(int)
        frames = frames[idx]
    elif len(frames) < num_frames:
        pad = np.repeat(frames[-1][None], num_frames-len(frames), axis=0)
        frames = np.concatenate([frames, pad], axis=0)

    frames = frames.astype(np.float32) / 255.0
    frames = torch.from_numpy(frames).permute(0,3,1,2)  # T,C,H,W

    return frames
