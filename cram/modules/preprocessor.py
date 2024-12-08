import jax.numpy as jnp
from cram.data.dataloader import load_shakespeare_dataset

def token_to_vector(token):
    max_bits = 20  # Fixed size to cover numbers up to 50,000
    binary_representation = f"{token:0{max_bits}b}"
    vector_representation = jnp.array([float(bit) for bit in binary_representation])
    return vector_representation


def compute_retention_matrix(train_ds,alpha):
    print("Size of training Dataset",len(train_ds))
    i = 0
    for t in train_ds:
        # if i == 10:
        #     break
        if i == 0:
            m_t = jnp.array([token_to_vector(t)])
            r_t = alpha*jnp.array([token_to_vector(t)])
        else:
            v = token_to_vector(t)
            r_i = alpha*(r_t[-1] + v)
            r_t = jnp.vstack([r_t,r_i])
            m_t = jnp.vstack([m_t,v])
        i = i + 1
        print(i)
    return r_t, m_t

if __name__ == '__main__':
    path = "/Users/abhinavmuraleedharan/cram/cram/data/input.txt"
    train_ds, test_ds = load_shakespeare_dataset(path)
    alpha = 0.5
    r_t,m_t = compute_retention_matrix(train_ds,alpha)
    temp_1 = alpha*(m_t[2] + alpha*m_t[1] + ((alpha)**2)*m_t[0])
    temp_2 = r_t[2]
    print(temp_2)
    print(temp_1)
    print(temp_2-temp_1)
    print(r_t)


