import jax.numpy as jnp
from cram.data.dataloader import load_shakespeare_dataset
from concurrent.futures import ThreadPoolExecutor

def token_to_vector(token):
    max_bits = 20  # Fixed size to cover numbers up to 50,000
    binary_representation = f"{token:0{max_bits}b}"
    vector_representation = jnp.array([float(bit) for bit in binary_representation])
    return vector_representation


def compute_vectors_parallel(tokens):
    """Compute vectors for tokens in parallel."""
    with ThreadPoolExecutor() as executor:
        vectors = list(executor.map(token_to_vector, tokens))
    return jnp.array(vectors)


def compute_retention_matrix(train_ds, alpha):
    print("Size of training Dataset", len(train_ds))
    # Compute all vectors in parallel
    print("Started Computation of Token Representations")
    vectors = compute_vectors_parallel(train_ds)
    print("Finished Computation of Token Representations")
    # Initialize retention matrix
    m_t = vectors
    r_t = jnp.zeros_like(m_t)

    # Compute the retention matrix iteratively
    for i in range(len(train_ds)):
        if i == 0:
            r_t = alpha * m_t
        else:
            r_t = jnp.vstack([r_t, alpha * (r_t[-1] + m_t[i])])

        if (i + 1) % 1000 == 0:  # Print every 1000 iterations for progress monitoring
            print(f"Processed {i + 1} tokens")
    
    return r_t, m_t


if __name__ == '__main__':
    path = "/Users/abhinavmuraleedharan/cram/cram/data/input.txt"
    train_ds, test_ds = load_shakespeare_dataset(path)
    alpha = 0.5
    r_t, m_t = compute_retention_matrix(train_ds, alpha)
    
    temp_1 = alpha * (m_t[2] + alpha * m_t[1] + (alpha ** 2) * m_t[0])
    temp_2 = r_t[2]
    print(temp_2)
    print(temp_1)
    print(temp_2 - temp_1)
    print(r_t)
