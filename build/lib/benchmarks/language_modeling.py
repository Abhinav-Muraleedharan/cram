# benchmarks/language_modeling.py
import jax
import jax.numpy as jnp
from typing import Dict, Optional
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

def evaluate_language_modeling(
    model,
    tokenizer,
    dataset_name: str = "wikitext-103-raw-v1",
    split: str = "validation",
    max_length: int = 2048,
    batch_size: int = 8
):
    # Load dataset
    dataset = load_dataset(dataset_name, split=split)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            max_length=max_length,
            truncation=True,
            return_tensors='np'
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Evaluation function
    @jax.jit
    def eval_step(params, batch):
        logits = model.apply({'params': params}, batch['input_ids'])
        log_probs = jax.nn.log_softmax(logits)
        
        # Calculate perplexity
        labels = batch['labels']
        mask = batch['attention_mask']
        
        loss = -jnp.sum(
            jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0] * mask
        ) / jnp.sum(mask)
        
        return {'loss': loss, 'perplexity': jnp.exp(loss)}
    
    # Run evaluation
    metrics = []
    for batch in tokenized_dataset:
        batch_metrics = eval_step(model.params, batch)
        metrics.append(batch_metrics)
    
    # Aggregate metrics
    final_metrics = {
        k: jnp.mean(jnp.stack([m[k] for m in metrics]))
        for k in metrics[0].keys()
    }