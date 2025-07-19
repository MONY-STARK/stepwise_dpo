import torch
from trl import DPOTrainer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class StepwiseDPOTrainer(DPOTrainer):
    """
    A custom DPOTrainer that incorporates step-level reward aggregation.
    It expects the dataset to contain 'chosen_step_scores' and 'rejected_step_scores'
    which are lists of scores for each step in the chosen/rejected responses.
    """

    @staticmethod
    def stepwise_collate_fn(batch):
        """
        Custom collate function for batching data in Stepwise DPOTrainer.
        It pads sequences and also handles padding for step-level scores.
        """
        def pad_tensor_field(key, dtype=torch.long, pad_val=0):
            tensors = [torch.tensor(sample[key], dtype=dtype) for sample in batch]
            return pad_sequence(tensors, batch_first=True, padding_value=pad_val)
        
        return {
            "prompt_input_ids": pad_tensor_field("prompt_input_ids"),
            "prompt_attention_mask": pad_tensor_field("prompt_attention_mask"),
            "chosen_input_ids": pad_tensor_field("chosen_input_ids"),
            "chosen_attention_mask": pad_tensor_field("chosen_attention_mask"),
            "rejected_input_ids": pad_tensor_field("rejected_input_ids"),
            "rejected_attention_mask": pad_tensor_field("rejected_attention_mask"),
            "chosen_step_scores": pad_tensor_field("chosen_step_scores", dtype=torch.float32, pad_val=0.0),
            "rejected_step_scores": pad_tensor_field("rejected_step_scores", dtype=torch.float32, pad_val=0.0),
            # Optionally pad step indices if you're using them later
            "step_indices_chosen": pad_tensor_field("step_indices_chosen") if "step_indices_chosen" in batch[0] else None,
            "step_indices_rejected": pad_tensor_field("step_indices_rejected") if "step_indices_rejected" in batch[0] else None,
        }

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=self.stepwise_collate_fn,
        )

    def compute_rewards(self, model_inputs: dict, chosen_step_scores: torch.Tensor, rejected_step_scores: torch.Tensor):
        """
        Computes the aggregate rewards for chosen and rejected responses by summing
        their respective step-level scores.
        """
        chosen_rewards = torch.tensor(
            [score[:len(score)].sum().item() for score in chosen_step_scores],
            dtype=torch.float32,
            device=chosen_step_scores.device
        )
        rejected_rewards = torch.tensor(
            [score[:len(score)].sum().item() for score in rejected_step_scores],
            dtype=torch.float32,
            device=rejected_step_scores.device
        )
        return chosen_rewards, rejected_rewards

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = super().compute_loss(model, inputs, return_outputs=True)

        chosen_step_scores = inputs["chosen_step_scores"]
        rejected_step_scores = inputs["rejected_step_scores"]

        chosen_rewards, rejected_rewards = self.compute_rewards(
            inputs, chosen_step_scores, rejected_step_scores
        )

        outputs['loss'] = self.dpo_loss(
            outputs["chosen_logps"],
            outputs["rejected_logps"],
            chosen_rewards,
            rejected_rewards,
        )

        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]
