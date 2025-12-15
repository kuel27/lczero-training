"""Normalization layers for the model."""

from typing import Type, Union

from flax import nnx

from proto import model_config_pb2


def get_norm_class(
    norm_type: int,
) -> Type[Union[nnx.RMSNorm, nnx.LayerNorm]]:
    """Get the normalization class based on the configuration.

    Args:
        norm_type: The normalization type from the config.

    Returns:
        The normalization class to use.
    """
    if norm_type == model_config_pb2.NormalizationType.RMS_NORM:
        return nnx.RMSNorm
    # Default to LayerNorm (including LAYER_NORM and unset)
    return nnx.LayerNorm
