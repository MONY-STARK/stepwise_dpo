# /home/mony/stepwise_dpo/Deliverables/Deliverables-2/prepare_dataset.py

from datasets import Dataset
from transformers import PreTrainedTokenizer

# Align max_length with the model's maximum input length (TinyLlama is typically 2048)
# This function prepares the raw text into tokenized IDs and attention masks.
# The actual truncation for the DPO trainer happens based on DPOConfig's max_length.
# However, it's good practice to pre-tokenize with a reasonable max_length to avoid
# very large initial tensors.
def tokenize_function(example, tokenizer: PreTrainedTokenizer, max_length=2048):
    """
    Tokenizes a single example for DPO training, including prompt, chosen, and rejected texts,
    and also passes through step-level scores.

    Args:
        example (dict): A dictionary containing 'prompt', 'chosen', 'rejected',
                        and optionally 'chosen_step_scores' and 'rejected_step_scores'.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        max_length (int): The maximum sequence length for tokenization.
                          Should not exceed the model's maximum context window.

    Returns:
        dict: A dictionary of tokenized inputs and step scores, ready for DPO.
    """
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]

    # Tokenize separately with truncation and padding.
    # The 'truncation=True' will cut sequences longer than max_length.
    # The 'padding="max_length"' will pad shorter sequences to max_length.
    # This prepares the data for efficient batching and consumption by the trainer.
    prompt_tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length)
    chosen_tokens = tokenizer(prompt + "\n" + chosen, truncation=True, padding="max_length", max_length=max_length)
    rejected_tokens = tokenizer(prompt + "\n" + rejected, truncation=True, padding="max_length", max_length=max_length)

    return {
        "prompt_input_ids": prompt_tokens["input_ids"],
        "prompt_attention_mask": prompt_tokens["attention_mask"],
        "chosen_input_ids": chosen_tokens["input_ids"],
        "chosen_attention_mask": chosen_tokens["attention_mask"],
        "rejected_input_ids": rejected_tokens["input_ids"],
        "rejected_attention_mask": rejected_tokens["attention_mask"],
        # Ensure these keys exist in your dataset or provide defaults
        "chosen_step_scores": example.get("chosen_step_scores", [1.0]),
        "rejected_step_scores": example.get("rejected_step_scores", [0.0]),
    }

def prepare_stepwise_dpo_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizer):
    """
    Prepares a dataset for Stepwise DPO training by tokenizing it.

    Args:
        dataset (Dataset): The raw dataset containing prompt, chosen, rejected, and step scores.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.

    Returns:
        Dataset: The tokenized dataset.
    """
    print("🔹 Tokenizing dataset...")
    # Pass the tokenizer and a consistent max_length (2048 for TinyLlama)
    # The `batched=False` is fine for smaller datasets or if examples vary widely in length.
    return dataset.map(lambda x: tokenize_function(x, tokenizer, max_length=2048), batched=False)