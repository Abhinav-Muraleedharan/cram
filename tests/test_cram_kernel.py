import pytest
import jax.numpy as jnp
from flax.linen import Module
from jax.random import PRNGKey
import jax
from cram.models.cram import CRAMKernel  # Replace `your_module` with the correct module name

@pytest.fixture
def random_input():
    """Fixture to generate random input data."""
    seq_length = 10
    feature_size = 8
    hidden_size = 8
    rng_key = PRNGKey(0)
    x = jax.random.normal(rng_key, (seq_length, feature_size))
    return x, hidden_size

def test_kernel_initialization(random_input):
    """Test initialization of the CRAMKernel."""
    x, hidden_size = random_input
    kernel = CRAMKernel(hidden_size=hidden_size)
    params = kernel.init(PRNGKey(1), x, training=True)
    assert "kernel_layer" in params["params"], "Kernel layer is not initialized properly."
   
def test_kernel_output_shape(random_input):
    """Test the output shape of the kernel."""
    x, hidden_size = random_input
    print("Input Shape:", x.shape)
    kernel = CRAMKernel(hidden_size=hidden_size)
    params = kernel.init(PRNGKey(1), x, training=True)
    y_out = kernel.apply(params, x, training=True)
    print("Output:",y_out)
    assert y_out.shape[0] == x.shape[0], "Output shape is not as expected."

def test_kernel_output_values(random_input):
    """Test the numerical behavior of the kernel."""
    x, hidden_size = random_input
    kernel = CRAMKernel(hidden_size=hidden_size)
    params = kernel.init(PRNGKey(1), x, training=True)
    y_out = kernel.apply(params, x, training=True)
    assert jnp.all(y_out >= 0) and jnp.all(y_out <= 1), "Output is not in the range [0, 1]."
    assert not jnp.isnan(y_out).any(), "Output contains NaN values."


def test_kernel_batch_dimension(random_input):
    """Test the kernel behavior with a different batch dimension."""
    seq_length = 6
    batch_size = 10  # Use a larger batch size for this test
    feature_size = 8
    hidden_size = 8
    rng_key = PRNGKey(42)
    x = jax.random.normal(rng_key, (batch_size,seq_length,feature_size))
    
    kernel = CRAMKernel(hidden_size=hidden_size)
    params = kernel.init(PRNGKey(1), x, training=True)
    y_out = kernel.apply(params, x, training=True)
    print("Batch Output",y_out)
    print("Batch Output Shape:",y_out.shape)

    # Check if the output has the correct batch dimension and feature size
    assert y_out.shape[0] == batch_size, "Output batch size does not match input batch size."
    assert y_out.shape[1] == feature_size, "Output feature size does not match input feature size."
    print("Batch Dimension Test Passed: Output Shape:", y_out.shape)


def test_kernel_training_flag(random_input):
    """Test the training flag in the kernel."""
    x, hidden_size = random_input
    kernel = CRAMKernel(hidden_size=hidden_size)
    params = kernel.init(PRNGKey(1), x, training=True)
    y_out_training = kernel.apply(params, x, training=True)
    y_out_inference = kernel.apply(params, x, training=False)
    # Depending on implementation, output should be identical or differ if training logic is added.
    assert jnp.allclose(y_out_training, y_out_inference), "Training flag affects output unexpectedly."
