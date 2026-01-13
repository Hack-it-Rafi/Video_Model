import random

def split_by_recording(folders, ratio=0.8):
    random.shuffle(folders)
    k=int(len(folders)*ratio)
    return folders[:k], folders[k:]