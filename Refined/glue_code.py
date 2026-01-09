import pandas as pd

def create_llm_input(csv_path):
    df = pd.read_csv(csv_path)
    
    def create_desc(row):
        if pd.notna(row['attribute_value']):
            return f"{row['full_label']} ({row['attribute_value']})"
        return row['full_label']

    df['description'] = df.apply(create_desc, axis=1)

    timeline_events = []
    current_event = None
    start_time_sec = 0
    duration = 0
    chunk_len = 5 

    for i, row in df.iterrows():
        event = row['description']
        
        if event != current_event:
            if current_event is not None:
                end_time_sec = start_time_sec + duration
                timeline_events.append(
                    f"[{start_time_sec//60}:{start_time_sec%60:02d}-"
                    f"{end_time_sec//60}:{end_time_sec%60:02d}] {current_event}"
                )
            
            current_event = event
            start_time_sec = i * chunk_len
            duration = chunk_len
        else:
            duration += chunk_len

    if current_event:
        end_time_sec = start_time_sec + duration
        timeline_events.append(
            f"[{start_time_sec//60}:{start_time_sec%60:02d}-"
            f"{end_time_sec//60}:{end_time_sec%60:02d}] {current_event}"
        )
        
    return "\n".join(timeline_events)

if __name__ == "__main__":
    text_data = create_llm_input("annotations.csv")
    with open("condensed_timeline.txt", "w") as f:
        f.write(text_data)
    print("condensed_timeline.txt created successfully.")