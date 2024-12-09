import optax
import pytest
import jax.numpy as jnp
from typing import Any, Dict
from cram.modules.loss import cross_entropy_loss, mse_loss

class MockModel:
    """Mock model class for testing purposes."""
    def apply(self, params: Dict[str, Any], batch: jnp.ndarray, training: bool) -> jnp.ndarray:
        """Simulate model prediction."""
        # Simply return the params['weight'] * batch for testing
        out = jnp.dot(batch, params['weight'])
        #out = batch
        return out

@pytest.fixture
def setup_mse_test_data():
    """Fixture to set up test data for MSE loss function."""
    # Create simple test data
    batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    targets = jnp.array([3.0, 7.0])
    params = {'weight': jnp.array([1.0, 1.0])}
    model = MockModel()
    
    return batch, targets, params, model

@pytest.fixture
def setup_cross_entropy_test_data():
    """Fixture to set up test data for cross-entropy loss function."""
    # Create simple test data for classification
    batch = jnp.array([[100.0, 2.0 ,4.4], [3.0, 0.0,13.5]])
    targets = jnp.array([0, 2])  # Binary classification targets
    params = {'weight': jnp.array([[1.0, -1.0,2.9], [1.0, 1.,2.3],[1.0, 1.,2.3]])}
    model = MockModel()
    
    return batch, targets, params, model

def test_mse_loss_zero():
    """Test MSE loss when predictions exactly match targets."""
    batch = jnp.array([[1.0]])
    targets = jnp.array([2.0])
    params = {'weight': jnp.array([2.0])}
    model = MockModel()
    
    loss = mse_loss(params, model, batch, targets)
    assert jnp.allclose(loss, 0.0)

def test_mse_loss_nonzero(setup_mse_test_data):
    """Test MSE loss with known difference between predictions and targets."""
    batch, targets, params, model = setup_mse_test_data
    
    loss = mse_loss(params, model, batch, targets)
    # Expected predictions: [3.0, 7.0]
    # Expected loss: mean((3-3)^2 + (7-7)^2) = 0
    assert jnp.allclose(loss, 0.0)

def test_mse_loss_shape(setup_mse_test_data):
    """Test that MSE loss returns a scalar."""
    batch, targets, params, model = setup_mse_test_data
    
    loss = mse_loss(params, model, batch, targets)
    assert loss.ndim == 0

def test_cross_entropy_loss_shape(setup_cross_entropy_test_data):
    """Test that cross-entropy loss returns a scalar."""
    batch, targets, params, model = setup_cross_entropy_test_data
    
    loss = cross_entropy_loss(params, model, batch, targets)
    assert loss.ndim == 0

def test_cross_entropy_loss_range(setup_cross_entropy_test_data):
    """Test that cross-entropy loss is non-negative."""
    batch, targets, params, model = setup_cross_entropy_test_data
    
    loss = cross_entropy_loss(params, model, batch, targets)
    print("Loss:", loss)
    assert loss >= 0

def test_cross_entropy_loss_perfect_prediction():
    """Test cross-entropy loss with perfect predictions."""
    batch = jnp.array([[1.0]])
    targets = jnp.array([1])
    params = {'weight': jnp.array([[0.0, 1000.0]])}  # Very confident prediction
    model = MockModel()
    
    loss = cross_entropy_loss(params, model, batch, targets)
    assert jnp.allclose(loss, 0.0, atol=1e-3)