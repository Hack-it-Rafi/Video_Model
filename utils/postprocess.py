import itertools
import json

def merge_actions(sequence):
    merged = []
    for label, group in itertools.groupby(sequence):
        count = len(list(group))
        duration = count * 5
        if "idle" in label or "reading" in label and duration < 30:
            continue
        merged.append({"action": label.replace(":", " ").title(), "duration_sec": duration})
    return merged