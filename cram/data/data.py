
import torch
import numpy as np
import jax.numpy as jnp
from pydantic import BaseModel
from typing import Dict, Iterator, Callable
from torch.utils.data import DataLoader, Dataset
from cram.models.cram import CRAM, CRAMConfig
from itertools import islice
from datasets import load_dataset
from transformers import AutoTokenizer

    
class DataConfig(BaseModel):
    d_pos: int = 16
    vocab_size: int = 50257  # GPT-2 vocab size
    seq_len: int = 128
    batch_size: int = 4

class ChunkedDataset(Dataset):
    """Dataset class for chunked tokens"""
    def __init__(self, chunks: list, prepare_fn: Callable):
        self.chunks = chunks
        self.prepare_fn = prepare_fn
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        prepared = self.prepare_fn(chunk)
        return {k: torch.tensor(v) for k, v in prepared.items()}

class WikiTextDataset:
    def __init__(
        self,
        config: CRAMConfig,
        split: str = "train",
        dataset_name: str = "wikitext-103-raw-v1",
        tokenizer_name: str = "gpt2",
    ):
        self.config = config
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        
        # Load dataset
        print(f"Loading {dataset_name} dataset...")
        self.dataset = load_dataset("Salesforce/wikitext",dataset_name, split=split)
        
        # Load tokenizer
        print(f"Loading {tokenizer_name} tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Preprocess dataset
        print("Tokenizing dataset...")
        self.tokenized_dataset = self.dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing",
        )
        
    def _tokenize_function(self, examples):
        """Tokenize text examples"""
        outputs = self.tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
            return_attention_mask=True,
        )
        return outputs
    
    def _create_position_ids(self, seq_len: int) -> np.ndarray:
        """Create position embeddings for CRAM"""
        # Create binary position representations
        positions = np.arange(seq_len)
        position_ids = np.zeros((seq_len, self.config.d_pos))
        
        for i in range(seq_len):
            # Convert position to binary and pad to d_pos length
            binary = format(positions[i], f'0{self.config.d_pos}b')
            position_ids[i] = np.array([int(b) for b in binary])
            
        return position_ids
    
    def _prepare_chunk(self, chunk_tokens: list) -> Dict[str, np.ndarray]:
        """Prepare a single chunk of tokens as model inputs"""
        # Pad or truncate to seq_len
        if len(chunk_tokens) > self.seq_len:
            chunk_tokens = chunk_tokens[:self.seq_len]
        elif len(chunk_tokens) < self.seq_len:
            padding_length = self.seq_len - len(chunk_tokens)
            chunk_tokens.extend([self.tokenizer.pad_token_id] * padding_length)
        
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in chunk_tokens]
        
        # Create position ids
        position_ids = self._create_position_ids(self.seq_len)
        
        return {
            'input_ids': np.array(chunk_tokens),
            'position_ids': position_ids,
            'labels': np.array(chunk_tokens),  # For causal LM, labels are the same as inputs
            'attention_mask': np.array(attention_mask)
        }
    
    def _chunked_tokens(self) -> Iterator[list]:
        """Yield chunks of tokens of length seq_len"""
        current_chunk = []
        
        # Iterate through all tokenized examples
        for example in self.tokenized_dataset:
            tokens = example['input_ids']
            
            for token in tokens:
                current_chunk.append(token)
                if len(current_chunk) == self.seq_len:
                    yield current_chunk
                    current_chunk = []
            
            # Add document separator at the end of each example
            if current_chunk:
                current_chunk.append(self.tokenizer.eos_token_id)
        
        # Yield any remaining tokens
        if current_chunk:
            yield current_chunk
    
    def get_dataloader(self, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """Create PyTorch DataLoader that yields batches of data"""
        chunks = list(self._chunked_tokens())
        dataset = ChunkedDataset(chunks, self._prepare_chunk)
        
        def collate_fn(batch):
            """Convert batch of PyTorch tensors to JAX arrays"""
            batch_dict = {
                key: jnp.array([item[key].numpy() for item in batch])
                for key in batch[0].keys()
            }
            return batch_dict
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )

def create_train_dataloader(config: CRAMConfig, num_workers: int = 0) -> DataLoader:
    """Create training data loader with WikiText dataset"""
    dataset = WikiTextDataset(config, split="train")
    return dataset.get_dataloader(shuffle=True, num_workers=num_workers)

def create_val_dataloader(config: CRAMConfig, num_workers: int = 0) -> DataLoader:
    """Create validation data loader with WikiText dataset"""
    dataset = WikiTextDataset(config, split="validation")
    return dataset.get_dataloader(shuffle=False, num_workers=num_workers)

# Example usage:
if __name__ == "__main__":
    # Test the dataloader  
    config = DataConfig()
    
    # Create dataloaders with no multiprocessing for testing
    print("\nCreating dataloaders...")
    train_loader = create_train_dataloader(config, num_workers=0)
    val_loader = create_val_dataloader(config, num_workers=0)
    
    # Test a few batches
    print("\nTesting training dataloader:")
    for batch_idx, batch in enumerate(islice(train_loader, 2)):
        print(f"\nBatch {batch_idx}:")
        for k, v in batch.items():
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
    
    print("\nTesting validation dataloader:")
    for batch_idx, batch in enumerate(islice(val_loader, 2)):
        print(f"\nBatch {batch_idx}:")
        for k, v in batch.items():
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")