import torch
import torch.nn as nn
import numpy as np
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Union, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SEED = 42


@dataclass
class ABMILUCBTrainingArguments(TrainingArguments):
    """Training arguments for ABMIL UCB trainer."""
    num_features: int = 196
    ucb_enabled: bool = True
    ucb_beta: float = 1.0
    ucb_warmup_iter: int = 500
    ucb_top_k: int = 10
    n_heads: int = 8
    n_branches: int = 1


class ABMILFeatureDataset(Dataset):
    """Dataset for ABMIL with variable-length features and target embeddings."""
    
    def __init__(self, features_list, targets_list, num_features=196):
        self.features_list = features_list
        self.targets_list = targets_list
        self.num_features = num_features
        
    def __len__(self):
        return len(self.features_list)
        
    def __getitem__(self, idx):
        features = torch.tensor(self.features_list[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets_list[idx], dtype=torch.float32)
        
        # Sample features with deterministic strategy
        num_available = features.shape[0]
        if num_available >= self.num_features:
            indices = torch.randperm(num_available, generator=torch.Generator().manual_seed(SEED))[:self.num_features]
        else:
            indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(SEED))
        
        sampled_features = features[indices]
        
        return {
            'features': sampled_features,
            'targets': targets,
            'labels': targets
        }


class ABMILUCBTrainer(Trainer):
    """Trainer for ABMIL with UCB patch selection mechanism."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ucb_step_counter = 0
        self.ucb_count_score = None
        self.best_accuracy = 0.0
        self.train_losses = []
        self.eval_losses = []
        
    def _initialize_ucb_count_score(self, batch_size: int, num_images: int):
        """Initialize UCB count score tensor for ABMIL."""
        if self.ucb_count_score is None:
            # Shape: (batch_size, num_images, n_branches, n_heads)
            self.ucb_count_score = torch.zeros(
                batch_size,
                num_images,
                self.args.n_branches,
                self.args.n_heads,
                device=self.args.device,
                dtype=torch.float32
            )
            logger.info(f"Initialized UCB count score: {self.ucb_count_score.shape}")
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss computation for ABMIL with UCB."""
        features = inputs.get('features')
        targets = inputs.get('targets')
        if targets is None:
            targets = inputs.get('labels')
        
        if targets is None:
            raise ValueError("Neither 'targets' nor 'labels' found in inputs")
        
        batch_size, num_images, feature_dim = features.shape
        
        # Initialize UCB count score if needed
        if self.args.ucb_enabled and self.ucb_count_score is None:
            self._initialize_ucb_count_score(batch_size, num_images)
        
        # Increment step counter
        self.ucb_step_counter += 1
        
        # Forward pass with UCB parameters
        aggregated_features, head_attentions, updated_count_score = model(
            features=features,
            counter=self.ucb_step_counter,
            UCB_Count_Score=self.ucb_count_score,
            ucb=self.args.ucb_enabled,
            ucb_beta=self.args.ucb_beta,
            ucb_warmup_iter=self.args.ucb_warmup_iter,
            ucb_top_k=self.args.ucb_top_k
        )
        
        # Update UCB count score
        if self.args.ucb_enabled:
            self.ucb_count_score = updated_count_score
        
        # Compute similarity loss between aggregated features and targets
        loss = self.compute_similarity_loss(aggregated_features, targets)
        
        outputs = {
            'aggregated_features': aggregated_features,
            'head_attentions': head_attentions,
            'loss': loss
        }
        
        return (loss, outputs) if return_outputs else loss
    
    def compute_similarity_loss(self, aggregated_features, targets):
        """Compute cosine similarity loss between features and targets."""
        # Handle multiple branches by taking mean or using first branch
        if aggregated_features.dim() == 3:  # (batch_size, n_branches, feature_dim)
            if aggregated_features.shape[1] == 1:
                embeddings = aggregated_features.squeeze(1)
            else:
                embeddings = aggregated_features.mean(dim=1)  # Average across branches
        else:
            embeddings = aggregated_features
            
        # Compute cosine similarity loss
        cos_sim = nn.functional.cosine_similarity(embeddings, targets, dim=1)
        loss = 1 - cos_sim.mean()
        
        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """Custom prediction step for ABMIL evaluation."""
        features = inputs.get('features')
        targets = inputs.get('targets')
        if targets is None:
            targets = inputs.get('labels')
        
        if targets is None:
            raise ValueError("Neither 'targets' nor 'labels' found in inputs")
        
        model.eval()
        
        with torch.no_grad():
            aggregated_features, head_attentions, _ = model(
                features=features,
                counter=self.ucb_step_counter,
                UCB_Count_Score=self.ucb_count_score,
                ucb=self.args.ucb_enabled,
                ucb_beta=self.args.ucb_beta,
                ucb_warmup_iter=self.args.ucb_warmup_iter,
                ucb_top_k=self.args.ucb_top_k
            )
            
            loss = self.compute_similarity_loss(aggregated_features, targets)
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Return aggregated features as predictions
        predictions = aggregated_features.squeeze(1) if aggregated_features.dim() == 3 else aggregated_features
        return (loss, predictions, targets)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Custom evaluation with performance tracking."""
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        eval_loss = eval_results.get(f'{metric_key_prefix}_loss', 0.0)
        self.eval_losses.append(eval_loss)
        
        eval_accuracy = eval_results.get(f'{metric_key_prefix}_accuracy', 0.0)
        if eval_accuracy > self.best_accuracy:
            self.best_accuracy = eval_accuracy
            logger.info(f"New best accuracy: {self.best_accuracy:.5f}")
        
        return eval_results
    
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Custom logging for training loss tracking."""
        super().log(logs, *args, **kwargs)
        
        if 'train_loss' in logs:
            self.train_losses.append(logs['train_loss'])
    
    def save_training_history(self, output_dir: str):
        """Save training history and UCB statistics."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'train_losses.npy'), np.array(self.train_losses))
        np.save(os.path.join(output_dir, 'eval_losses.npy'), np.array(self.eval_losses))
        
        # Save UCB count score statistics
        if self.ucb_count_score is not None:
            ucb_stats = {
                'mean_counts': self.ucb_count_score.mean().item(),
                'std_counts': self.ucb_count_score.std().item(),
                'max_counts': self.ucb_count_score.max().item(),
                'min_counts': self.ucb_count_score.min().item()
            }
            np.save(os.path.join(output_dir, 'ucb_stats.npy'), ucb_stats)
        
        logger.info(f"Training history saved to {output_dir}")


class ABMILUCBCallback(TrainerCallback):
    """Callback for ABMIL UCB-specific monitoring."""
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            trainer = kwargs.get('trainer')
            if trainer and hasattr(trainer, 'ucb_step_counter'):
                warmup_status = "active" if trainer.ucb_step_counter > args.ucb_warmup_iter else "warmup"
                logger.info(f"Step {state.global_step}, UCB counter: {trainer.ucb_step_counter}, Status: {warmup_status}")
    
    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs.get('trainer')
        if trainer and hasattr(trainer, 'ucb_count_score') and trainer.ucb_count_score is not None:
            mean_usage = trainer.ucb_count_score.mean().item()
            logger.info(f"Evaluation at step {state.global_step}, Mean UCB usage: {mean_usage:.3f}")


def compute_similarity_metric(eval_pred: EvalPrediction):
    """Compute similarity metric for evaluation."""
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    
    # Compute cosine similarity
    cos_sim = nn.functional.cosine_similarity(
        torch.tensor(predictions), 
        torch.tensor(labels), 
        dim=1
    )
    
    accuracy = cos_sim.mean().item()
    
    return {
        "accuracy": accuracy,
        "mean_similarity": accuracy,
        "std_similarity": cos_sim.std().item()
    }


def create_abmil_ucb_trainer(
    model,
    training_args: ABMILUCBTrainingArguments,
    train_dataset: ABMILFeatureDataset,
    eval_dataset: Optional[ABMILFeatureDataset] = None,
    callbacks: Optional[List[TrainerCallback]] = None
) -> ABMILUCBTrainer:
    """Factory function to create ABMIL UCB trainer."""
    if callbacks is None:
        callbacks = []
    
    callbacks.append(ABMILUCBCallback())
    
    trainer = ABMILUCBTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_similarity_metric,
        callbacks=callbacks
    )
    
    return trainer




if __name__ == "__main__":
    pass