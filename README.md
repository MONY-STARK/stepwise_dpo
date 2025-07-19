# 📘 stepwise_dpo

`stepwise_dpo` is an **extended version of Direct Preference Optimization (DPO)** that integrates fine-grained feedback by using a secondary LLM as a **Generative Reward Model (GenRM)**. 

Instead of assigning a single reward to a full output (as in standard DPO), this approach **rewards each step** of a generated multi-step solution, aggregates the rewards, and uses them in DPO-style training.

---

## 💡 What is Stepwise DPO?

Stepwise DPO introduces step-level granularity to preference optimization:

- A second LLM (called **GenRM**) evaluates each individual step of a response.
- Step-level scores (typically between -1 and 1) are assigned to both *chosen* and *rejected* outputs.
- The final reward for each output is calculated by aggregating its step scores.
- These aggregated rewards guide the learning of the main model using the DPO loss.

This makes the reward signal more detailed and aligned with human-style reasoning.

---

## ✅ Steps Taken in Deliverable 1

1. **Downloaded an open-source LLM** to act as GenRM  
   - I used `zephyr-7b-beta`, a capable and lightweight open-source model.
   - It worked well in assigning meaningful scores to reasoning steps.

2. **Formatted the recommended dataset `PRM800K`**  
   - This dataset contains math word problems and step-by-step solutions.
   - Reformatted samples into:
     - `prompt`
     - `chosen` (preferred solution)
     - `rejected` (less preferred solution)
     - `chosen_step_scores` (per-step reward by GenRM)
     - `rejected_step_scores`

3. **Used the selected model to generate stepwise scores**  
   - Parsed each solution into individual reasoning steps.
   - Ran each step through the GenRM (Zephyr-7B) to assign scores.
   - Saved the results in `stepwise_dpo_dataset.json`.

---

## ✅ Deliverable 2 — Stepwise DPO Training

In Deliverable 2, we train a reward-optimized language model using the step-level rewards generated from Deliverable 1.

### 🔧 Key Steps

1. **Customized DPO Trainer**  
   A new `StepwiseDPOTrainer` was implemented by subclassing `DPOTrainer`.  
   - Handled padding and collation of variable-length `step_scores`.
   - Aggregated per-step rewards using summation.
   - Integrated aggregated rewards into the DPO loss computation.

2. **Data Preparation**  
   The dataset was tokenized and structured to include:
   - `prompt_input_ids`, `chosen_input_ids`, `rejected_input_ids`
   - `chosen_step_scores`, `rejected_step_scores`  
   This preserved the structure needed for reward aggregation.

3. **Training Loop**  
   The model was trained with the aggregated rewards to prefer `chosen` completions over `rejected` ones, using the DPO loss adapted for stepwise scoring.

### 🎯 Outcome

This enables training a preference model that not only selects the better solution but does so by learning from **step-level reasoning quality**.