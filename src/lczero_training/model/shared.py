import math

import jax
import jax.nn
from flax import nnx
from flax.linen import initializers as flax_initializers

from proto import net_pb2

from .utils import get_activation


def compute_swiglu_hidden_size(mlp_hidden: int, multiple: int = 128) -> int:
    """Compute the SwiGLU hidden size to match parameter count with standard MLP.

    SwiGLU uses 3 projections (gate, up, down) vs standard FFN's 2 projections.
    For equal parameters: 3 * d * h_swiglu = 2 * d * h_mlp
    Therefore: h_swiglu = (2/3) * h_mlp

    Args:
        mlp_hidden: The hidden size of the standard MLP to match against.
        multiple: Round up to this multiple for hardware efficiency.

    Returns:
        Hidden size for SwiGLU that approximately matches MLP parameter count.
    """
    # (2/3) * mlp_hidden, rounded up to multiple
    hidden = math.ceil(2 * mlp_hidden / 3)
    return ((hidden + multiple - 1) // multiple) * multiple


class Ffn(nnx.Module):
    """Feed-forward network with support for standard MLP and SwiGLU variants.

    Standard MLP (MISH, SWISH, etc.):
        linear1: d -> hidden
        activation
        linear2: hidden -> d

    SwiGLU (ACTIVATION_SWIGLU):
        gate_proj: d -> hidden
        up_proj: d -> hidden
        down_proj: hidden -> d
        forward: down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_activation: net_pb2.NetworkFormat.ActivationFunction,
        deepnorm_beta: float,
        *,
        rngs: nnx.Rngs,
    ):
        deepnorm_init = flax_initializers.variance_scaling(
            scale=deepnorm_beta,
            mode="fan_avg",
            distribution="truncated_normal",
        )
        out_features = in_features
        self.activation = hidden_activation
        self.is_swiglu = (
            hidden_activation == net_pb2.NetworkFormat.ACTIVATION_SWIGLU
        )

        if self.is_swiglu:
            # SwiGLU: 3 projections (gate, up, down)
            self.gate_proj = nnx.Linear(
                in_features=in_features,
                out_features=hidden_features,
                kernel_init=deepnorm_init,
                rngs=rngs,
            )
            self.up_proj = nnx.Linear(
                in_features=in_features,
                out_features=hidden_features,
                kernel_init=deepnorm_init,
                rngs=rngs,
            )
            self.down_proj = nnx.Linear(
                in_features=hidden_features,
                out_features=out_features,
                kernel_init=deepnorm_init,
                rngs=rngs,
            )
        else:
            # Standard 2-layer MLP
            self.linear1 = nnx.Linear(
                in_features=in_features,
                out_features=hidden_features,
                kernel_init=deepnorm_init,
                rngs=rngs,
            )
            self.linear2 = nnx.Linear(
                in_features=hidden_features,
                out_features=out_features,
                kernel_init=deepnorm_init,
                rngs=rngs,
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.is_swiglu:
            # SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
            gate = jax.nn.silu(self.gate_proj(x))
            up = self.up_proj(x)
            return self.down_proj(gate * up)
        else:
            # Standard MLP
            x = self.linear1(x)
            x = get_activation(self.activation)(x)
            x = self.linear2(x)
            return x
