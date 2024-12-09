"""
Implementation of Cumulative Retention Autoregressive Model (CRAM)

asda 
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Optional, Tuple
from functools import partial

class CRAMEmbeddings(nn.Module):
    """Embeddings module for CRAM"""
    vocab_size: int
    hidden_size: int
    max_position_embeddings: int
    dropout_rate: float = 0.1
    
    def setup(self):
        self.word_embeddings = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            name='word_embeddings'
        )
        
        self.position_embeddings = nn.Embed(
            num_embeddings=self.max_position_embeddings,
            features=self.hidden_size,
            name='position_embeddings'
        )
        
        self.layer_norm = nn.LayerNorm(epsilon=1e-5)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
    
    def __call__(self, input_ids: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        input_shape = input_ids.shape
        seq_length = input_shape[1]
        
        # Get word embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        
        # Create position IDs and embeddings
        position_ids = jnp.arange(seq_length)[None, :]
        position_embeddings = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, deterministic=not training)
        
        return embeddings

class CRAMBlock(nn.Module):
    """Single CRAM block with retention mechanism"""
    hidden_size: int
    intermediate_size: int = None
    dropout_rate: float = 0.1
    
    def setup(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        
        self.retention_layer = nn.Dense(self.hidden_size)
        self.layer_norm1 = nn.LayerNorm(epsilon=1e-5)
        self.layer_norm2 = nn.LayerNorm(epsilon=1e-5)
        
        # FFN layers
        self.ffn = nn.Sequential([
            nn.Dense(self.intermediate_size),
            nn.gelu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate)
        ])
    
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Compute retention
        def retention_step(carry, x_t):
            xi_prev = carry
            xi_t = jax.nn.sigmoid(self.retention_layer(x_t))
            xi_new = 0.5 * xi_t + 0.5 * xi_prev
            return xi_new, xi_new
        
        # Initialize retention state
        init_state = jnp.zeros((x.shape[0], self.hidden_size))
        
        # Apply retention mechanism
        _, retention_seq = jax.lax.scan(
            retention_step,
            init_state,
            jnp.transpose(x, (1, 0, 2))
        )
        
        retention_output = jnp.transpose(retention_seq, (1, 0, 2))
        
        # First residual connection
        hidden_states = self.layer_norm1(x + retention_output)
        
        # FFN and second residual connection
        ffn_output = self.ffn(hidden_states, deterministic=not training)
        hidden_states = self.layer_norm2(hidden_states + ffn_output)
        
        return hidden_states

class CRAMPreTrainedModel(nn.Module):
    """Base class for CRAM models"""
    config: Any
    dtype: jnp.dtype = jnp.float32
    
    def _init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> Any:
        raise NotImplementedError()

class CRAMModel(CRAMPreTrainedModel):
    """Main CRAM model implementation"""
    
    def setup(self):
        self.embeddings = CRAMEmbeddings(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            max_position_embeddings=self.config.max_position_embeddings,
            dropout_rate=self.config.dropout_rate
        )
        
        self.layers = [
            CRAMBlock(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                dropout_rate=self.config.dropout_rate
            )
            for _ in range(self.config.num_hidden_layers)
        ]
        
        self.final_layer_norm = nn.LayerNorm(epsilon=1e-5)
        
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        training: bool = True,
        output_hidden_states: bool = False,
    ) -> Any:
        hidden_states = self.embeddings(input_ids, training=training)
        
        all_hidden_states = () if output_hidden_states else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            hidden_states = layer(hidden_states, training=training)
        
        hidden_states = self.final_layer_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            return hidden_states, all_hidden_states
        
        return hidden_states

class CRAMForCausalLM(CRAMPreTrainedModel):
    """CRAM model with language modeling head"""
    
    def setup(self):
        self.transformer = CRAMModel(self.config)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
    
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Any:
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            training=training
        )
        
        hidden_states = transformer_outputs[0] if isinstance(transformer_outputs, tuple) else transformer_outputs
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def generate(
        self,
        input_ids: jnp.ndarray,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """Generate text using the model"""
        
        def sample_token(logits: jnp.ndarray, temperature: float, rng: jax.random.PRNGKey) -> jnp.ndarray:
            if temperature == 0:
                return jnp.argmax(logits, axis=-1)
            
            logits = logits / temperature
            
            # Top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
                logits = jnp.zeros_like(logits).at[top_k_indices].set(top_k_logits)
            
            # Top-p sampling
            if top_p < 1.0:
                sorted_logits = jnp.sort(logits, axis=-1)[::-1]
                cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits), axis=-1)
                mask = cumulative_probs <= top_p
                mask = jnp.concatenate([jnp.ones_like(mask[:1]), mask[:-1]], axis=0)
                sorted_logits = jnp.where(mask, sorted_logits, -float('inf'))
                logits = jnp.zeros_like(logits).at[jnp.argsort(logits)[::-1]].set(sorted_logits)
            
            # Sample from distribution
            return jax.random.categorical(rng, logits)
        
        def generate_step(carry, _):
            current_ids, rng = carry
            
            # Get logits for next token
            logits = self(current_ids, training=False)
            next_token_logits = logits[:, -1, :]
            
            # Sample next token
            rng, sample_rng = jax.random.split(rng)
            next_token = sample_token(next_token_logits, temperature, sample_rng)
            
            # Update sequence
            new_ids = jnp.concatenate([current_ids, next_token[:, None]], axis=1)
            
            return (new_ids, rng), next_token
        
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        # Generate tokens one at a time
        init_carry = (input_ids, rng)
        (final_ids, _), _ = jax.lax.scan(
            generate_step,
            init_carry,
            None,
            length=max_length - input_ids.shape[1]
        )
        
        return final_ids 