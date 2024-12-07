import argparse
import jax.numpy as jnp 
from datasets import load_dataset
from transformers import GPT2Tokenizer


#ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def load_wikitext_dataset():
    train_dataset = load_dataset("Salesforce/wikitext","wikitext-103-raw-v1", split="train")
    val_dataset = load_dataset("Salesforce/wikitext","wikitext-103-raw-v1", split="validation")
    test_dataset  = load_dataset("Salesforce/wikitext","wikitext-103-raw-v1", split="test")
    return train_dataset, val_dataset 

def load_shakespeare_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    data = jnp.array(tokenizer.encode(text))
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    print(train_data)
    val_data = data[n:]
    return train_data, val_data

if __name__ == '__main__':
    print("Vocabulary size:", tokenizer.vocab_size)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset Name.")
    parser.add_argument("--path", type=str, help ="Path to Dataset")
    args = parser.parse_args()
    if args.dataset == 'wikitext':
        train_ds, val_ds = load_wikitext_dataset()
    else:
        train_ds, val_ds = load_shakespeare_dataset()
    