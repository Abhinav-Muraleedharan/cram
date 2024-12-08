import jax
import jax.numpy as jnp
import optax
from typing import Any, Optional
from functools import partial
from cram.models.cram_simple import ResidualNN, create_model, forward_pass

def mse_loss(params, model, batch, targets):
    """Calculate Mean Squared Error loss."""
    predictions = model.apply(params, batch)
    return jnp.mean((predictions - targets) ** 2)

#@jax.jit
def train_step(params, opt_state, model, batch, targets, optimizer):
    """Perform a single training step.
    
    Args:
        params: Model parameters
        opt_state: Optimizer state
        model: ResidualNN instance
        batch: Input batch
        targets: Target values
        optimizer: Optax optimizer
    
    Returns:
        Tuple of (updated params, updated optimizer state, loss)
    """
    loss_grad_fn = jax.value_and_grad(mse_loss)
    loss, grads = loss_grad_fn(params, model, batch, targets)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def train_model(model, params, train_data, train_targets, 
                n_epochs: int = 100, batch_size: int = 32, 
                learning_rate: float = 1e-3):
    """Train the model.
    
    Args:
        model: ResidualNN instance
        params: Initial model parameters
        train_data: Training data of shape (n_samples, n_in)
        train_targets: Training targets of shape (n_samples, n_out)
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Returns:
        Final model parameters
    """
    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Training loop
    n_batches = len(train_data) // batch_size
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        
        # Shuffle data
        perm = jax.random.permutation(jax.random.PRNGKey(epoch), len(train_data))
        train_data_shuffled = train_data[perm]
        train_targets_shuffled = train_targets[perm]
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = batch_start + batch_size
            
            batch = train_data_shuffled[batch_start:batch_end]
            targets = train_targets_shuffled[batch_start:batch_end]
            
            # Train step
            params, opt_state, loss = train_step(
                params, opt_state, model, batch, targets, optimizer)
            total_loss += loss
            
        avg_loss = total_loss / n_batches
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    return params

if __name__ == "__main__":
    # Example usage
    
    # Generate some dummy data
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    
    n_samples = 1000
    n_in = 20
    n_hidden = 32
    n_out = 20
    
    # Random input data
    X = jax.random.normal(key, (n_samples, n_in))
    # Random target data
    Y = jax.random.normal(subkey, (n_samples, n_out))
    
    # Create and train model
    model, params = create_model(n_in, n_hidden, n_out, key)
    
    # Train the model
    final_params = train_model(model, params, X, Y)
    
    # Make predictions
    test_input = jax.random.normal(key, (1, n_in))
    prediction = forward_pass(final_params, model, test_input)
    print("Test prediction shape:", prediction.shape)