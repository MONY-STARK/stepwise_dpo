import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOConfig

# Assuming 'StepwiseDPOTrainer' and 'prepare_stepwise_dpo_dataset' are in these files
from trainer import StepwiseDPOTrainer
from prepare_dataset import prepare_stepwise_dpo_dataset

# --- Config ---
# Ensure these models/paths are valid for your setup
model_id = "tiny_llama"  # Use an actual valid HuggingFace repo ID, e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer_model_id = "tiny_llama" # Use an actual valid HuggingFace repo ID for tokenizer
data_path = "Deliverables/Deliverables-2/stepwise_dpo_dataset.json" # Path to your dataset
output_dir = "./stepwise-dpo-output"

# --- Load Model ---
print(f"🔹 Loading model: {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16, # Use bf16 if supported and preferred
    device_map="auto" # Automatically maps model to available devices (e.g., GPU)
)

# --- Load Tokenizer ---
print(f"🔹 Loading tokenizer: {tokenizer_model_id}...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id)
# Ensure pad_token is set, often eos_token is a good fallback
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"❗ tokenizer.pad_token was None, set to tokenizer.eos_token ({tokenizer.eos_token_id})")

# --- Load and Prepare Dataset ---
print(f"🔹 Loading dataset from: {data_path}...")
# Assuming your JSON file contains a list of dictionaries with "prompt", "chosen", "rejected",
# "chosen_step_scores", and "rejected_step_scores"
dataset = load_dataset("json", data_files={"train": data_path})["train"]

# Tokenize the dataset using our custom preparation function
tokenized_dataset = prepare_stepwise_dpo_dataset(dataset, tokenizer)

# --- Training Config ---
training_args = DPOConfig(
    output_dir="./dpo_results", # Or your desired output directory
    per_device_train_batch_size=4, # Adjust based on your GPU memory
    gradient_accumulation_steps=1, # Adjust as needed
    learning_rate=5e-5,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    report_to="none", # Or "wandb", "tensorboard" etc.
    # --- ADD OR UPDATE THIS LINE ---
    max_length=2048, # This is crucial: Truncate sequences to model's max length
    # max_prompt_length=512, # You might also want to control prompt length separately
    # You might also need to set max_target_length if your dataset has very long responses
    # which get concatenated. For TinyLlama, 2048 is its full context window.

    # Ensure padding_value matches your tokenizer's pad_token_id
    padding_value=tokenizer.pad_token_id, # Keep this as it is
    truncation_mode="keep_end", # Common setting for DPO
    # other DPO-specific args like beta, etc.
)

# --- Initialize Trainer ---
print("🔹 Initializing StepwiseDPOTrainer...")
trainer = StepwiseDPOTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    # eval_dataset=tokenized_eval_dataset, # Uncomment and prepare if you have an evaluation set
)

# --- Train ---
print("🚀 Training started...")
trainer.train()

print("✅ Training complete!")

# You might want to save the final model explicitly if save_steps doesn't capture the end
# trainer.save_model(f"{output_dir}/final_model")