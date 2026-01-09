import ollama

def generate_workflow(timeline_text):
    """
    Feeds the timeline to Ollama (Mistral) and gets a structured summary.
    """
    
    system_prompt = (
        "You are a Workflow Analysis AI. "
        "I will provide a log of screen actions with timestamps. "
        "Your goal is to summarize this into a user story. "
        "Merge repetitive actions (like typing) into single steps. "
        "Format: 'Step 1: User did X...'"
    )
    
    user_prompt = f"Here is the action log:\n{timeline_text}\n\nSummarize the workflow:"
    
    try:
        response = ollama.chat(model='mistral', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ])
        
        return response['message']['content']
        
    except Exception as e:
        return f"Error connecting to Ollama: {e}. Make sure the Ollama app is running."

if __name__ == "__main__":
    try:
        with open("condensed_timeline.txt", "r") as f:
            timeline_data = f.read()
            
        print("Asking Local LLM (Ollama)...")
        result = generate_workflow(timeline_data)
        print("\n--- FINAL WORKFLOW ---")
        print(result)
    except FileNotFoundError:
        print("condensed_timeline.txt not found. Run the glue code first.")