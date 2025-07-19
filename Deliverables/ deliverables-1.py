from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from Data_formatting import format_data
import re

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_id = "zephyr-7b-beta"  # Ensure this is correct
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

# Load and format the dataset
formatted_data = format_data("/home/mony/stepwise_dpo/prm800k/prm800k/data/phase1_train.jsonl")
print(f"Length of Data: {len(formatted_data)}")

# Process the first item
# item = formatted_data[0]
for question, steps in formatted_data[0].items():
    for i, step in enumerate(steps):
        print(f"\n🔹 Step {i + 1}")

        prompt = f"""You are a helpful math tutor.

        The student is solving the problem:
        "{question}"

        Here is the student's step:
        "{step}"

        Rate this step on a scale of -1 to 1 based on:
        - Correctness
        - Usefulness toward solving the problem
        - Clarity

        Just return the score (-1, 0, or 1):"""

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate output
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )

        # Decode and extract score
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Model Raw Response:", response)

        # Try to extract score from response
        match = re.search(r"\b-?1\b|\b0\b|\b1\b", response)
        score = int(match.group()) if match else None
        print("Parsed Score:", score)
