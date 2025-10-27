"""
Common model architectures and building blocks.

This module provides reusable neural network components:
- Encoders and decoders (MLP, CNN, RNN)
- Attention mechanisms
- Vector quantization layers
- Common architectural patterns
"""

from .encoders import MLPEncoder, ConvEncoder, RNNEncoder
from .decoders import MLPDecoder, ConvDecoder, RNNDecoder
from .vector_quantizer import VectorQuantizer

__all__ = [
    'MLPEncoder',
    'ConvEncoder',
    'RNNEncoder',
    'MLPDecoder',
    'ConvDecoder',
    'RNNDecoder',
    'VectorQuantizer',
]
