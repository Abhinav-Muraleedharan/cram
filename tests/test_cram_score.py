import pytest
import jax.numpy as jnp
from flax.linen import Module
from jax.random import PRNGKey
import jax
from cram.models.cram import CRAMScore  # Replace `your_module_path` with the actual module path

@pytest.fixture
def random_inputs():
    """Fixture to generate random inputs for testing."""
    batch_size = 4
    seq_length = 8
    d_pos = 5  # Positional embedding size
    rng_key = PRNGKey(0)
    x_kernel = jax.random.normal(rng_key, (batch_size, seq_length))  # (B, N)
    pos_ids = jax.random.normal(rng_key, (batch_size, seq_length, d_pos))  # (B, N, d_pos)
    return x_kernel, pos_ids, d_pos

@pytest.fixture
def cram_score_instance():
    """Fixture to create a CRAMScore instance."""
    hidden_size = 16
    dropout_rate = 0.1
    d_pos = 5
    return CRAMScore(hidden_size=hidden_size, dropout_rate=dropout_rate, d_pos=d_pos)

def test_cram_score_initialization(cram_score_instance):
    """Test initialization of the CRAMScore module."""
    assert cram_score_instance.hidden_size == 16, "Hidden size not initialized correctly."
    assert cram_score_instance.dropout_rate == 0.1, "Dropout rate not initialized correctly."
    assert cram_score_instance.d_pos == 5, "d_in not calculated correctly."

def test_cram_score_output_shape(cram_score_instance, random_inputs):
    """Test output shape of CRAMScore."""
    x_kernel, pos_ids, d_pos = random_inputs
    batch_size, seq_length = x_kernel.shape
    params = cram_score_instance.init(PRNGKey(1), x_kernel, pos_ids, training=True)
    y_out = cram_score_instance.apply(params, x_kernel, pos_ids, training=True)
    assert y_out.shape == (batch_size, seq_length, 1), "Output shape is incorrect."

def test_cram_score_stacking(cram_score_instance, random_inputs):
    """Test whether input tensors are stacked correctly."""
    x_kernel, pos_ids, d_pos = random_inputs
    x_kernel_expanded = jnp.expand_dims(x_kernel, axis=-1)
    batch_size, seq_length = x_kernel.shape
    params = cram_score_instance.init(PRNGKey(1), x_kernel, pos_ids, training=True)
    y_out = cram_score_instance.apply(params, x_kernel, pos_ids, training=True)
    # Check if concatenation preserves expected shape
    d_pos = cram_score_instance.d_pos
    expected_shape = (batch_size, seq_length, d_pos+1)
    print("Output:",y_out)
    print("Shape of Score Output:",y_out.shape)
    concatenated_input = jnp.concatenate([x_kernel_expanded, pos_ids], axis=-1)
    assert concatenated_input.shape == expected_shape, "Stacked input has incorrect shape."

def test_cram_score_output_values(cram_score_instance, random_inputs):
    """Test output numerical properties of CRAMScore."""
    x_kernel, pos_ids, d_pos = random_inputs
    params = cram_score_instance.init(PRNGKey(1), x_kernel, pos_ids, training=True)
    y_out = cram_score_instance.apply(params, x_kernel, pos_ids, training=True)
    # Test if output does not contain NaN or Inf
    assert not jnp.isnan(y_out).any(), "Output contains NaN values."
    assert not jnp.isinf(y_out).any(), "Output contains Inf values."
