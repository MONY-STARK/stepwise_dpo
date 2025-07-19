import json

def read_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    return dataset

def format_data(data_path):
    formatted_dataset = []
    dataset = read_data(data_path)
    
    for data in dataset[:3]:  # Remove slice to process full dataset
        question = data['question']["problem"].replace("\\$", "$")  # Clean LaTeX-style $ signs
        steps = []
        completions = data['label']['steps']
        
        for step in completions:
            chosen_idx = step.get("chosen_completion")
            if chosen_idx is not None:
                completions_list = step.get("completions", [])
                if 0 <= chosen_idx < len(completions_list):
                    step_text = completions_list[chosen_idx]["text"]
                    steps.append(step_text.strip())  # Optional: clean up whitespace

        formatted_dataset.append({question: steps})
    
    return formatted_dataset
