import yaml
from sklearn.model_selection import train_test_split
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, Trainer, TrainingArguments
from utils.dataset import ScreenActionDataset
from utils.collator import collator
import pandas as pd
import torch

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

df = pd.read_csv(cfg["data"]["annotations_path"])
unique_labels = df["full_label"].unique().tolist()
label2id = {l: i for i, l in enumerate(unique_labels)}
id2label = {v: k for k, v in label2id.items()}

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["full_label"], random_state=42)

processor = VideoMAEImageProcessor.from_pretrained(cfg["training"]["model_name"])
model = VideoMAEForVideoClassification.from_pretrained(
    cfg["training"]["model_name"],
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
)

train_dataset = ScreenActionDataset(cfg["data"]["annotations_path"], cfg["data"]["chunks_dir"], label2id)
val_dataset = ScreenActionDataset(cfg["data"]["annotations_path"], cfg["data"]["chunks_dir"], label2id)

train_dataset.df = train_df.reset_index(drop=True)
val_dataset.df = val_df.reset_index(drop=True)

args = TrainingArguments(
    output_dir=cfg["training"]["output_model_dir"],
    per_device_train_batch_size=cfg["training"]["batch_size"],
    per_device_eval_batch_size=cfg["training"]["batch_size"],
    num_train_epochs=cfg["training"]["epochs"],
    learning_rate=cfg["training"]["learning_rate"],
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
)

trainer.train()
trainer.save_model(cfg["training"]["output_model_dir"])
print("Training complete! Model saved.")