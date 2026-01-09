import yaml
from utils.postprocess import merge_actions
from openai import OpenAI

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

client = OpenAI()  

def summarize_workflow(merged_actions, output_path):
    prompt = f"""Convert this sequence into a clear numbered user workflow.
    Remove boring parts. Make it readable.

    {json.dumps(merged_actions, indent=2)}

    Example:
    1. User opened VSCode and typed code for 2 minutes
    2. Switched to browser and watched videos
    3. A ransomware note appeared
    """
    resp = client.chat.completions.create(
        model=cfg["inference"]["llm_model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    text = resp.choices[0].message.content
    with open(output_path, "w") as f:
        f.write(text)
    print(text)