import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision
from pathlib import Path
import numpy as np
from typing import Optional
import os

# Set cache directory to local folder instead of C drive
CACHE_DIR = Path(__file__).parent / 'model_cache'
CACHE_DIR.mkdir(exist_ok=True)
os.environ['TORCH_HOME'] = str(CACHE_DIR)

# Use torchvision video model r3d_18 (3D ResNet-18) — available in torchvision
# torchvision version includes video models (>=0.8+ usually)

class ChunkFeatureExtractor:
    def __init__(self, device: str = 'cuda', target_frames: int = 50, image_size: int = 112):
        self.device = device
        self.target_frames = target_frames
        self.image_size = image_size
        self.model = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.DEFAULT)
        
        if hasattr(self.model, 'fc'):
            in_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()
            self.feature_dim = in_dim
        else:
            self.feature_dim = 512
        self.model = self.model.to(self.device).eval()

        self.frame_transform = T.Compose([
            T.ConvertImageDtype(torch.float),
            T.Resize((self.image_size, self.image_size)),
            T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ])

    def _read_video_to_tensor(self, path: Path) -> torch.Tensor:
        from torchvision.io import read_video
        video_frames, _, info = read_video(str(path), pts_unit='sec')
        if video_frames.numel() == 0:
            raise RuntimeError(f"No frames read from {path}")
        video_frames = video_frames.permute(0, 3, 1, 2)  # T, C, H, W
        video_frames = video_frames / 255.0
        return video_frames

    def _temporal_resample(self, frames: torch.Tensor) -> torch.Tensor:
        T_in = frames.shape[0]
        if T_in == self.target_frames:
            return frames
        if T_in > 1:
            idx = np.linspace(0, T_in - 1, self.target_frames).astype(int)
            return frames[idx]
        else:
            return frames.repeat(self.target_frames, 1, 1, 1)

    def extract(self, chunk_path: Path) -> np.ndarray:
        """Read the whole chunk video, resample/pad to target_frames, run through 3D CNN and return 1D feature vector."""
        frames = self._read_video_to_tensor(chunk_path)  # T x C x H x W (float in 0..1)
        frames = self._temporal_resample(frames)
        proc_frames = []
        for f in frames:
            # f: C x H x W float in 0..1; need to make PIL? but transforms expect tensor
            # our transform uses ConvertImageDtype and Resize which accept tensors in newer torchvision
            proc = self.frame_transform(f)
            proc_frames.append(proc)
        # stack -> T x C x H x W, convert to C x T x H x W for video models
        x = torch.stack(proc_frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(self.device)  # 1 x C x T x H x W
        with torch.no_grad():
            feat = self.model(x)  # shape 1 x feature_dim (because fc->Identity)
        feat = feat.cpu().numpy().reshape(-1)
        return feat


if __name__ == '__main__':
    # quick test (requires a sample chunk video path)
    import sys
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('chunk_000.mp4')
    ext = ChunkFeatureExtractor(device='cpu', target_frames=50)
    v = ext.extract(p)
    print('feature dim', v.shape)