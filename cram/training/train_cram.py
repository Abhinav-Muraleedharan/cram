import time
import jax
import yaml
import argparse
import jax.numpy as jnp
import optax
from flax.training import train_state
from cram.models.cram import CRAM, CRAMConfig
from typing import Tuple, Dict, Any
from tqdm.auto import tqdm
import wandb
import pandas as pd
from pathlib import Path
import datetime

class TrainState(train_state.TrainState):
    """Custom train state for CRAM model"""
    batch_stats: Dict[str, Any]

def create_train_state(rng: jax.random.PRNGKey, config: CRAMConfig) -> TrainState:
    """Initialize training state"""
    model = CRAM(config)
    
    # Create dummy input for initialization
    batch_size = config.batch_size
    seq_length = config.seq_len
    position_ids = jnp.zeros((batch_size, seq_length, config.d_pos))
    input_ids = jnp.zeros((batch_size, seq_length), dtype=jnp.int32)
    
    # Initialize parameters
    variables = model.init(rng, input_ids, position_ids)
    
    # Create optimizer
    learning_rate = 1e-4  # Adjust as needed
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=0.01
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        batch_stats=variables.get('batch_stats', {})
    )

def compute_loss(logits: jnp.ndarray, labels: jnp.ndarray, padding_mask: jnp.ndarray) -> jnp.ndarray:
    """Compute cross entropy loss with padding mask"""
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    shift_padding_mask = padding_mask[:, 1:]
    
    # Compute cross entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        shift_logits, shift_labels
    )
    
    # Apply padding mask
    loss = loss * shift_padding_mask
    
    # Average loss over non-padded tokens
    return jnp.sum(loss) / jnp.sum(shift_padding_mask)

@jax.jit
def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    dropout_rng: jax.random.PRNGKey
) -> Tuple[TrainState, Dict[str, float]]:
    """Single training step"""
    input_ids = batch['input_ids']
    position_ids = batch['position_ids']
    labels = batch['labels']
    padding_mask = batch['attention_mask']
    
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            input_ids,
            position_ids,
            training=True,
            rngs={'dropout': dropout_rng}
        )
        loss = compute_loss(logits, labels, padding_mask)
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    # Update model
    state = state.apply_gradients(grads=grads)
    
    metrics = {
        'loss': loss,
        'perplexity': jnp.exp(loss)
    }
    
    return state, metrics

def train_epoch(
    state: TrainState,
    train_dataloader,
    rng: jax.random.PRNGKey,
    epoch: int
) -> Tuple[TrainState, Dict[str, float], list]:
    """Train for a single epoch"""
    batch_metrics = []
    step_metrics = []  # Store metrics for each step
    
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}')):
        rng, dropout_rng = jax.random.split(rng)
        state, metrics = train_step(state, batch, dropout_rng)
        
        # Convert JAX arrays to regular floats for logging
        step_metric = {
            'epoch': epoch,
            'batch': batch_idx,
            'loss': float(metrics['loss']),
            'perplexity': float(metrics['perplexity'])
        }
        
        # Log to wandb
        wandb.log(step_metric)
        
        batch_metrics.append(metrics)
        step_metrics.append(step_metric)
    
    # Compute mean of metrics across batches
    epoch_metrics = {
        k: float(jnp.mean(jnp.stack([metrics[k] for metrics in batch_metrics])))
        for k in batch_metrics[0].keys()
    }
    
    return state, epoch_metrics, step_metrics

def train(
    config: CRAMConfig,
    train_dataloader,
    experiment_name: str = None,
    num_epochs: int = 10,
    seed: int = 42
) -> TrainState:
    """Main training loop with logging"""
    # Initialize wandb
    if experiment_name is None:
        experiment_name = f"cram_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project="cram-training",
        name=experiment_name,
        config={
            "model_config": config.model_dump(),
            "num_epochs": num_epochs,
            "seed": seed
        }
    )
    
    # Create directory for saving metrics
    save_dir = Path(f"logs/{experiment_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    
    # Initialize training state
    state = create_train_state(init_rng, config)
    
    # Lists to store all metrics
    all_epoch_metrics = []
    all_step_metrics = []
    
    # Training loop
    for epoch in range(num_epochs):
        rng, epoch_rng = jax.random.split(rng)
        start_time = time.time()
        
        state, epoch_metrics, step_metrics = train_epoch(
            state, train_dataloader, epoch_rng, epoch
        )
        
        epoch_time = time.time() - start_time
        epoch_metrics['epoch'] = epoch
        epoch_metrics['time'] = epoch_time
        
        # Log epoch metrics
        wandb.log({f"epoch_{k}": v for k, v in epoch_metrics.items()})
        
        print(f'Epoch {epoch + 1}: loss = {epoch_metrics["loss"]:.4f}, '
              f'perplexity = {epoch_metrics["perplexity"]:.4f}, '
              f'time = {epoch_time:.2f}s')
        
        all_epoch_metrics.append(epoch_metrics)
        all_step_metrics.extend(step_metrics)
    
    # Save metrics to CSV
    pd.DataFrame(all_epoch_metrics).to_csv(
        save_dir / "epoch_metrics.csv", index=False
    )
    pd.DataFrame(all_step_metrics).to_csv(
        save_dir / "step_metrics.csv", index=False
    )
    
    # Close wandb run
    wandb.finish()
    
    return state

# Example usage:
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train the CRAM model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for the training run"
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        config = CRAMConfig.model_validate(config_dict)
    
    # Create dummy data loader for demonstration
    # Replace this with your actual data loading logic
    def create_dummy_dataloader(config):
        while True:
            yield {
                'input_ids': jnp.zeros((config.batch_size, config.seq_len), dtype=jnp.int32),
                'position_ids': jnp.zeros((config.batch_size, config.seq_len, config.d_pos)),
                'labels': jnp.zeros((config.batch_size, config.seq_len), dtype=jnp.int32),
                'attention_mask': jnp.ones((config.batch_size, config.seq_len))
            }
    
    train_dataloader = create_dummy_dataloader(config)
    
    # Train the model
    trained_state = train(config, train_dataloader, experiment_name=args.experiment_name)