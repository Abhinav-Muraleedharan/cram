import jax.numpy as jnp
from cram.data.dataloader import load_shakespeare_dataset

path = "/Users/abhinavmuraleedharan/cram/cram/data/input.txt"
train_ds, test_ds = load_shakespeare_dataset(path)

i = 0
alpha = 0.8

def token_to_vector(token):
    max_bits = 20  # Fixed size to cover numbers up to 50,000
    binary_representation = f"{token:0{max_bits}b}"
    vector_representation = jnp.array([int(bit) for bit in binary_representation])
    return vector_representation

for t in train_ds:
    if i == 1000:
        break
    print(i)
    if i == 0:
        m_t = token_to_vector(t)
    else:
        v = token_to_vector(t)
        m_t = jnp.vstack([m_t,v])
    i = i + 1
print(m_t)
