from typing import Any, Callable

import jax.numpy as jnp
from flax import nnx
from jax.nn import mish

from proto import net_pb2
from proto.hlo_pb2 import XlaShapeProto

# Mapping of pointwise activation functions.
# Note: ACTIVATION_SWIGLU is NOT included here because it is not a pointwise
# activation function - it changes the FFN structure and is handled specially
# in the Ffn class (shared.py).
_POINTWISE_ACTIVATIONS: dict[
    net_pb2.NetworkFormat.ActivationFunction, Callable[..., Any]
] = {
    net_pb2.NetworkFormat.ACTIVATION_MISH: mish,
    net_pb2.NetworkFormat.ACTIVATION_RELU: nnx.relu,
    net_pb2.NetworkFormat.ACTIVATION_NONE: lambda x: x,
    net_pb2.NetworkFormat.ACTIVATION_TANH: nnx.tanh,
    net_pb2.NetworkFormat.ACTIVATION_SIGMOID: nnx.sigmoid,
    net_pb2.NetworkFormat.ACTIVATION_SELU: nnx.selu,
    net_pb2.NetworkFormat.ACTIVATION_SWISH: nnx.swish,
    net_pb2.NetworkFormat.ACTIVATION_SOFTMAX: nnx.softmax,
}


def get_activation(
    activation: net_pb2.NetworkFormat.ActivationFunction,
) -> Callable[..., Any]:
    """Get a pointwise activation function.

    Args:
        activation: The activation function enum value.

    Returns:
        The corresponding activation function.

    Raises:
        ValueError: If ACTIVATION_SWIGLU is passed, since it's not pointwise.
        KeyError: If an unknown activation is passed.
    """
    if activation == net_pb2.NetworkFormat.ACTIVATION_SWIGLU:
        raise ValueError(
            "ACTIVATION_SWIGLU is not a pointwise activation function. "
            "It changes the FFN structure and should only be used as "
            "ffn_activation in DefaultsConfig, not passed to get_activation(). "
            "The Ffn class handles SWIGLU specially."
        )
    return _POINTWISE_ACTIVATIONS[activation]


def get_dtype(dtype: XlaShapeProto.Type) -> jnp.dtype:
    return {
        XlaShapeProto.PRED: jnp.bool_,
        XlaShapeProto.S4: jnp.int4,
        XlaShapeProto.S8: jnp.int8,
        XlaShapeProto.S16: jnp.int16,
        XlaShapeProto.S32: jnp.int32,
        XlaShapeProto.S64: jnp.int64,
        XlaShapeProto.U4: jnp.uint4,
        XlaShapeProto.U8: jnp.uint8,
        XlaShapeProto.U16: jnp.uint16,
        XlaShapeProto.U32: jnp.uint32,
        XlaShapeProto.U64: jnp.uint64,
        XlaShapeProto.F16: jnp.float16,
        XlaShapeProto.F32: jnp.float32,
        XlaShapeProto.BF16: jnp.bfloat16,
        XlaShapeProto.F64: jnp.float64,
        XlaShapeProto.F8E5M2: jnp.float8_e5m2,
        XlaShapeProto.F8E4M3FN: jnp.float8_e4m3fn,
        XlaShapeProto.F8E4M3B11FNUZ: jnp.float8_e4m3b11fnuz,
        XlaShapeProto.F8E5M2FNUZ: jnp.float8_e5m2fnuz,
        XlaShapeProto.F8E4M3FNUZ: jnp.float8_e4m3fnuz,
        XlaShapeProto.C64: jnp.complex64,
        XlaShapeProto.C128: jnp.complex128,
    }[dtype]
