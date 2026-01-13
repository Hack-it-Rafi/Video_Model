import torch, json
from smoothing import smooth_probs

def predict(model, clips, maps, window):
    model.eval(); pa,pp=[],[]
    with torch.no_grad():
        for v in clips:
            a,p = model(v.unsqueeze(0))
            pa.append(torch.softmax(a,1).cpu().numpy()[0])
            pp.append(torch.softmax(p,1).cpu().numpy()[0])
    pa_s = smooth_probs(pa,window); pp_s = smooth_probs(pp,window)
    results=[]
    for i,(a,p) in enumerate(zip(pa_s,pp_s)):
        ai=a.argmax(); pi=p.argmax()
        results.append({
            "chunk_id": f"chunk_{i:03d}",
            "action": maps['action_inv'][ai],
            "action_conf": float(a[ai]),
            "app": maps['app_inv'][pi],
            "app_conf": float(p[pi])
        })
    return results