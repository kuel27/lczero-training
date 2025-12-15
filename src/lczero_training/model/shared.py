import math

import jax
import jax.nn
from flax import nnx
from flax.linen import initializers as flax_initializers

from proto import net_pb2

from .utils import get_activation


def compute_swiglu_hidden_size(d: int, multiple: int = 128) -> int:
    """Compute the SwiGLU hidden size for parameter-matched FFN.

    SwiGLU uses 3 projections (gate, up, down) vs standard FFN's 2 projections.
    To match parameters: 3 * d * hidden_swiglu = 2 * d * hidden_mlp
    For hidden_mlp = 4d: hidden_swiglu = (8/3) * d

    Args:
        d: Model dimension (input/output size).
        multiple: Round up to this multiple for hardware efficiency.

    Returns:
        Hidden size rounded up to the specified multiple.
    """
    # (8/3) * d, rounded up to multiple
    hidden = math.ceil(8 * d / 3)
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
