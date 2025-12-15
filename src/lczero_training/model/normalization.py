"""Normalization layers for the model."""

from typing import Callable, Type, Union

import jax.numpy as jnp
from flax import nnx

from proto import model_config_pb2


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm is a simplification of LayerNorm that removes the mean centering
    operation, normalizing only by the root mean square of the input.

    This is computationally cheaper than LayerNorm while achieving similar
    or better results in many transformer architectures.

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        num_features: int,
        epsilon: float = 1e-3,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize RMSNorm.

        Args:
            num_features: Number of features in the input.
            epsilon: Small constant for numerical stability.
            rngs: Random number generators for parameter initialization.
        """
        del rngs  # Unused, but kept for API compatibility with LayerNorm
        self.scale = nnx.Param(jnp.ones((num_features,)))
        self.epsilon = epsilon

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply RMSNorm to the input.

        Args:
            x: Input tensor of shape (..., num_features).

        Returns:
            Normalized tensor of the same shape.
        """
        # Compute RMS: sqrt(mean(x^2) + epsilon)
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.epsilon)
        return (x / rms) * self.scale.value


def get_norm_class(
    norm_type: int,
) -> Type[Union[RMSNorm, nnx.LayerNorm]]:
    """Get the normalization class based on the configuration.

    Args:
        norm_type: The normalization type from the config.

    Returns:
        The normalization class to use.
    """
    if norm_type == model_config_pb2.NormalizationType.RMS_NORM:
        return RMSNorm
    # Default to LayerNorm (including LAYER_NORM and unset)
    return nnx.LayerNorm
