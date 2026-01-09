import os
import ffmpeg
import pandas as pd
from pathlib import Path

def chunk_all_videos(recordings_dir, chunks_dir):
    os.makedirs(chunks_dir, exist_ok=True)
    for video_file in Path(recordings_dir).glob("*.mp4"):
        out_pattern = str(chunks_dir / f"{video_file.stem}_chunk_%04d.mp4")
        try:
            (
                ffmpeg
                .input(str(video_file))
                .output(out_pattern, c='copy', segment_time=5, f='segment', reset_timestamps=1)
                .overwrite_output()
                .run(quiet=True)
            )
            print(f"Chunked {video_file.name}")
        except Exception as e:
            print(f"Error chunking {video_file.name}: {e}")

if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    chunk_all_videos(cfg["data"]["recordings_dir"], cfg["data"]["chunks_dir"])