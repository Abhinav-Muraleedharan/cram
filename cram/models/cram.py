"""
Implementation of Cumulative Retention Autoregressive Model (CRAM)

"""
import yaml
import jax
import argparse
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from pydantic import BaseModel, Field
from typing import Any, Optional, Tuple


class CRAMConfig(BaseModel):
    d_pos: int
    vocab_size: int
    n_layers: int
    d_hidden: int
    batch_size: int
    seq_len: int
    intermediate_size: int
    dropout_rate: float


def batch_binary_representation_matrix(batch_input_vectors, D):
    """
    Converts a batch of input vectors of integers into a batch of binary representation matrices.
    
    Args:
        batch_input_vectors (jnp.ndarray): Batch of input vectors (shape (B, N)).
        D (int): Number of bits for binary representation.
        
    Returns:
        jnp.ndarray: Tensor of shape (B, D, N), where each (D, N) slice corresponds to a binary representation matrix.
    """
    # batch_input_vectors shape: (B, N)
    B, N = batch_input_vectors.shape

    # Generate binary representations using broadcasting and bitwise operations
    binary_matrix = jnp.array(
        [(batch_input_vectors >> i) & 1 for i in range(D - 1, -1, -1)]
    )  # Shape: (D, B, N)

    # Rearrange to shape (B, D, N)
    binary_matrix = binary_matrix.transpose(1, 0, 2)  # Shape: (B, D, N)
    return binary_matrix




class CRAMEmbeddings(nn.Module):
    """Embeddings module for CRAM"""
    vocab_size: int
    hidden_size: int    
    def setup(self):
        self.word_embeddings = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            name='word_embeddings'
        )
        
    def __call__(self, input_ids: jnp.array, position_ids:jnp.array, training: bool = True) -> jnp.array:
        input_shape = input_ids.shape
        seq_length = input_shape[1]
        
        # Get word embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        # Combine embeddings
        embeddings = inputs_embeds 
        
        return embeddings, position_ids
    

class CRAMKernel(nn.Module):
    """ 
    Kernel Module for CRAM
    Input: x (B,N,D)
    Output: y_out (B,N) 
    """
    hidden_size: int
    def setup(self):
        self.kernel_layer = nn.Dense(self.hidden_size,use_bias=False)
    
    def __call__(self,x:jnp.array, training: bool= True) -> jnp.array:
        x_q = self.kernel_layer(x) # Implements Qx
        print("Shape of x_q:", x_q.shape)
        print("shape of x:",x.shape)
        y_out = jnp.sum(x * x_q,axis=1)
        print("Shape of y_out:", y_out.shape)
        y_out = nn.sigmoid(y_out)
        return y_out 



class CRAMScore(nn.Module):
    """ 
    N: Seq Length
    B: Batch Size

    Kernel Module for CRAM
    Input: x_kernel (B,N) & pos_ids (B,N)
    Output: y_out (B,N) 
    """
    hidden_size: int
    dropout_rate: int
    d_pos:int


    def setup(self):
        self.score_layer = nn.Sequential([
            nn.Dense(self.d_pos+1),
            nn.gelu,
            nn.Dense(self.hidden_size),
            nn.gelu,
            nn.Dense(1)
        ])
    
    def __call__(self,x_kernel:jnp.array, pos_ids:jnp.array, training: bool= True) -> jnp.array:
        # Assume pos_id is a preprocessed vector
        # first stack pos_id and x 
        # x_kernel (B,T)
        # pos_id (B,T,D)
        x_kernel_expanded = jnp.expand_dims(x_kernel, axis=-1) 
        x = jnp.concatenate([x_kernel_expanded, pos_ids], axis=-1) 
        print("shape of x:",x.shape)
        y_out = self.score_layer(x) # Implements f(x_i,i)
        print("Shape of y_out:", y_out.shape) # Expect: (B,T)
        return y_out 


#class RetentionLayer(nn.Module):


class RetentionLayer(nn.Module):
    """

    Computes the cumulative weighted sum for batched inputs:
    Y[b, :, i] = alpha_1 * x_1 + alpha_2 * x_2 + ... + alpha_i * x_i for each batch b.
    
    Args:
        X (jnp.ndarray): Array of shape (B, D, N), where each batch contains vectors x_i.
        s (jnp.ndarray): Array of shape (B, 1, N), where each batch contains scalars alpha_i.
    
    Returns:
        jnp.ndarray: Array of shape (B, D, N), cumulative weighted sums for each batch.
    """
     
    hidden_size:int
    
    def __call__(self, x:jnp.array, x_scores:jnp.array, training:bool=True):
        x_weighted = x_scores * x 
        y_out = jnp.cumsum(x_weighted, axis=-1) 
        return y_out

class CRAMBlock(nn.Module):
    """Single CRAM block with retention mechanism"""
    d_pos:int
    hidden_size: int
    intermediate_size: int = None
    dropout_rate: float = 0.1
    
    def setup(self):
        self.kernel_layer = CRAMKernel(self.hidden_size)
        self.score_layer = CRAMScore(hidden_size = self.hidden_size, dropout_rate= self.dropout_rate,d_pos=self.d_pos)
        self.retention_layer = RetentionLayer(hidden_size=self.hidden_size)
        self.layer_norm1 = nn.LayerNorm(epsilon=1e-5)
        self.layer_norm2 = nn.LayerNorm(epsilon=1e-5)
        
        # FFN layers
        self.ffn = nn.Sequential([
            nn.Dense(self.intermediate_size),
            nn.gelu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate)
        ])
    
    def __call__(self, x: jnp.ndarray, pos_ids:jnp.ndarray, training: bool = True) -> jnp.array:
        batch, seq_len, d_model = x.shape
        #shape of x: (batch, seq_len, d_model)
        #shape of position ids: (batch, seq_len)
        #1) compute kernel function 
        x_kernel = self.kernel_layer(x)
        x_scores = self.score_layer(x_kernel, pos_ids) # (batch, seq_len)
        x_retention = self.retention_layer(x_scores)

        retention_output = self.ffn(x_retention)
        # First residual connection
        hidden_states = self.layer_norm1(x_retention + retention_output)  
        return hidden_states


class CRAM(nn.Module):
    """Main CRAM model implementation"""
    config: Any 
    
    def setup(self):
        self.embeddings = CRAMEmbeddings(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size    
        )
        
        self.layers = [
            CRAMBlock(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                dropout_rate=self.config.dropout_rate,
                d_pos = self.config.d_pos
            )
            for _ in range(self.config.num_hidden_layers)
        ]
        
        self.final_layer_norm = nn.LayerNorm(epsilon=1e-5)
        
    def __call__(
        self,
        input_ids: jnp.array,
        position_ids: jnp.array,
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
    


def main(config:CRAMConfig):
    model = CRAM(config)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run the CRAM model.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        config = CRAMConfig.model_validate(config_dict)

    print(config.vocab_size)
    main(config)
   



# class CRAMForCausalLM(CRAMPreTrainedModel):
#     """CRAM model with language modeling head"""
    
#     def setup(self):
#         self.transformer = CRAMModel(self.config)
#         self.lm_head = nn.Dense(
#             self.config.vocab_size,
#             use_bias=False,
#             kernel_init=nn.initializers.normal(stddev=0.02)
#         )
    
#     def __call__(
#         self,
#         input_ids: jnp.ndarray,
#         attention_mask: Optional[jnp.ndarray] = None,
#         position_ids: Optional[jnp.ndarray] = None,
#         training: bool = True,
#     ) -> Any:
#         transformer_outputs = self.transformer(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             training=training
#         )
        
#         hidden_states = transformer_outputs[0] if isinstance(transformer_outputs, tuple) else transformer_outputs
        
#         # Project to vocabulary
#         logits = self.lm_head(hidden_states)
        
#         return logits
    
#     def generate(
#         self,
#         input_ids: jnp.ndarray,
#         max_length: int,
#         temperature: float = 1.0,
#         top_k: int = 50,
#         top_p: float = 0.9,
#         rng: Optional[jax.random.PRNGKey] = None,
#     ) -> jnp.ndarray:
#         """Generate text using the model"""
        
#         def sample_token(logits: jnp.ndarray, temperature: float, rng: jax.random.PRNGKey) -> jnp.ndarray:
#             if temperature == 0:
#                 return jnp.argmax(logits, axis=-1)
            
#             logits = logits / temperature
            
#             # Top-k sampling
#             if top_k > 0:
#                 top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
#                 logits = jnp.zeros_like(logits).at[top_k_indices].set(top_k_logits)
            
#             # Top-p sampling
#             if top_p < 1.0:
#                 sorted_logits = jnp.sort(logits, axis=-1)[::-1]
#                 cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits), axis=-1)
#                 mask = cumulative_probs <= top_p
#                 mask = jnp.concatenate([jnp.ones_like(mask[:1]), mask[:-1]], axis=0)
#                 sorted_logits = jnp.where(mask, sorted_logits, -float('inf'))
#                 logits = jnp.zeros_like(logits).at[jnp.argsort(logits)[::-1]].set(sorted_logits)
            
#             # Sample from distribution
#             return jax.random.categorical(rng, logits)
        
#         def generate_step(carry, _):
#             current_ids, rng = carry
            
#             # Get logits for next token
#             logits = self(current_ids, training=False)
#             next_token_logits = logits[:, -1, :]
            
#             # Sample next token
#             rng, sample_rng = jax.random.split(rng)
#             next_token = sample_token(next_token_logits, temperature, sample_rng)
            
#             # Update sequence
#             new_ids = jnp.concatenate([current_ids, next_token[:, None]], axis=1)
            
#             return (new_ids, rng), next_token
        
#         if rng is None:
#             rng = jax.random.PRNGKey(0)
        
#         # Generate tokens one at a time
#         init_carry = (input_ids, rng)
#         (final_ids, _), _ = jax.lax.scan(
#             generate_step,
#             init_carry,
#             None,
#             length=max_length - input_ids.shape[1]
#         )
        
#         return final_ids 