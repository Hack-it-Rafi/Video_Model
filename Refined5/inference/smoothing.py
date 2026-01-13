import numpy as np

def smooth_probs(probs, window):
    out=[]
    for i in range(len(probs)):
        s=max(0,i-window); e=min(len(probs),i+window+1)
        out.append(np.mean(probs[s:e],0))
    return out