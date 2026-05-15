"""Sequence models for time series prediction.

Architectures:
- Vanilla RNN (classic recurrent baseline)
- Transformer-based model (attention-based sequence model)
- Causal Transformer variant (pre-norm, learnable PE, causal attention,
  mean pooling) - used for the bonus transformer-variant comparison.
"""

from .rnn import VanillaRNN
from .transformer import TransformerModel
from .transformer_variant import CausalTransformerModel

__all__ = ['VanillaRNN', 'TransformerModel', 'CausalTransformerModel']
