import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
import ffmpeg
import os
from tqdm import tqdm
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

model = VideoMAEForVideoClassification.from_pretrained(cfg["training"]["output_model_dir"])
processor = VideoMAEImageProcessor.from_pretrained(cfg["training"]["output_model_dir"])
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def predict_video(video_path):
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    chunk_paths = []
    for start in range(0, int(duration), 5):
        out_path = f"temp_chunk_{start}.mp4"
        (
            ffmpeg.input(video_path, ss=start, t=5)
            .output(out_path, c='copy')
            .overwrite_output()
            .run(quiet=True)
        )
        chunk_paths.append(out_path)

    predictions = []
    for path in tqdm(chunk_paths, desc="Predicting"):
        stream = ffmpeg.input(path)
        video, _ = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True, quiet=True)
        import numpy as np
        frames = np.frombuffer(video, np.uint8).reshape(-1, 1080, 1920, 3)[:16]  # adjust resolution
        inputs = processor(list(frames), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = logits.argmax(-1).item()
        predictions.append(model.config.id2label[pred])
        os.remove(path)
    return predictions

if __name__ == "__main__":
    import sys
    preds = predict_video(sys.argv[1])
    print(preds[:20], "...")