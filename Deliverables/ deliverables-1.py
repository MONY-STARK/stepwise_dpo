from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json

model_id = "zephyr-7b-beta"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

llm = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# Load 10 samples from prm800k test split
data = []
with open("prm800k/prm800k/data/phase1_test.jsonl", "r") as f:
    for idx, line in enumerate(f):
        if idx == 10:
            break
        data.append(json.loads(line))

# Build prompt for each step
def build_prompt(question, step):
    return f"""You are a helpful math tutor.

The student is solving the problem:
"{question}"

Here is the student's step:
"{step}"

Rate this step on a scale of 1 to 5, based on:
- Correctness
- Usefulness toward solving the problem
- Clarity

Just return the score (1 to 5) and nothing else.
Answer:"""

# Function to extract numeric score
def extract_score(text):
    for token in text.strip():
        if token in "12345":
            return int(token)
    return -1

# Stepwise Evaluation Loop
results = []
for item in data:
    problem = item["problem"]
    steps = item["solution"]

    for idx, step in enumerate(steps):
        prompt = build_prompt(problem, step)
        response = llm(prompt, max_new_tokens=10, do_sample=False)
        generated = response[0]["generated_text"].strip()
        score = extract_score(generated)

        print(f"Step {idx+1}: Score={score}, Step='{step[:60]}...'")

        results.append({
            "problem": problem,
            "step": step,
            "score": score
        })

# Save results to JSON
with open("zephyr_stepwise_scores.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Saved Zephyr-based stepwise evaluation to 'zephyr_stepwise_scores.json'")
