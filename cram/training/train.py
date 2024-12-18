
import numpy as np
import time
import jax
import optax
import jax.numpy as jnp
from functools import partial
from typing import Any, Optional, Callable
from cram.data.dataloader import load_shakespeare_dataset, load_wikitext_dataset
from cram.models.cram_simple import ResidualNN, create_model, forward_pass
from transformers import GPT2Tokenizer


#ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def count_parameters(params):
    return sum(jnp.size(param) for param in jax.tree_util.tree_leaves(params))

def mse_loss(params, model, batch, targets):
    """Calculate Mean Squared Error loss."""
    predictions = model.apply(params, batch, training=True)
    return jnp.mean((predictions - targets) ** 2)

def cross_entropy_loss(params,model,batch,targets):
    predictions = model.apply(params,batch,training=True)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(predictions, targets))

def train_step(params, opt_state, model, batch, targets, tx):
    """Perform a single training step.
    
    Args:
        params: Model parameters
        opt_state: Optimizer state
        model: Flax model instance
        batch: Input batch
        targets: Target values
        tx: Optax optimizer
    
    Returns:
        Tuple of (updated params, updated optimizer state, loss)
    """
    loss_fn = lambda p: cross_entropy_loss(p, model, batch, targets)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Create a JIT-compiled version of the training step
@partial(jax.jit, static_argnums=(2,5))
def train_step_jit(params, opt_state, model, batch, targets, tx):
    return train_step(params, opt_state, model, batch, targets, tx)

def train_model(model, params, train_data, train_targets, 
                n_epochs: int = 100, batch_size: int = 8, 
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
    tx = optax.adam(learning_rate)
    opt_state = tx.init(params)
    
    # Training loop
    n_batches = len(train_data) // batch_size

    
    for epoch in range(n_epochs):
        total_loss = 0.0
        
        # Shuffle data
        # key = jax.random.PRNGKey(epoch)
        # perm = jax.random.permutation(key, len(train_data))
        # train_data_shuffled = train_data[perm]
        # train_targets_shuffled = train_targets[perm]
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = batch_start + batch_size
            
            batch = train_data[batch_start:batch_end]
            targets = jnp.array([train_targets[batch_start:batch_end]]).T
            targets = train_targets[batch_start+1:batch_end+1]
            # print("Batch Size:",batch.shape)
            # print("Targets Size:",targets.shape)
            
            # Train step
            params, opt_state, loss = train_step_jit(
                params, opt_state, model, batch, targets, tx)
            total_loss += loss
            
        avg_loss = total_loss / n_batches
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    return params

if __name__ == "__main__":
    d_vocab = tokenizer.vocab_size 
    # load dataset:
    r_t = np.load('/Users/abhinavmuraleedharan/cram/cram/modules/rt.npy')
    train_ds, val_ds = load_shakespeare_dataset('/Users/abhinavmuraleedharan/cram/cram/data/input.txt')
    # with open('/Users/abhinavmuraleedharan/cram/cram/modules/rt.npy', 'wb') as f:
    #     np.save(f, r_t)
    start_time = time.time()
    # Generate some dummy data
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    
    n_samples = 1000
    n_in = 20
    n_hidden = 32
    n_out = d_vocab
    print("train data:",train_ds[0:4])
    print("Size of training data:",len(train_ds))
    # Random input data
    X = r_t
    # Random target data
    Y = train_ds
    
    # Create and train model
    model, params = create_model(n_in, n_hidden, n_out, key)
    # Print the number of parameters
    print(f"Number of parameters: {count_parameters(params)}")
    # Train the model
    final_params = train_model(model, params, X, Y)
    
    # Make predictions
    test_input = jax.random.normal(key, (1, n_in))
    prediction = forward_pass(final_params, model, test_input)
    end_time = time.time()
    print("Test prediction shape:", prediction.shape)
    print("Total Execution time:", end_time-start_time)