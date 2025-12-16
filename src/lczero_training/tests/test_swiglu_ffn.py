"""Test SwiGLU FFN implementation."""

import jax
import jax.numpy as jnp
import jax.random
import pytest
from flax import nnx

from proto import net_pb2

from lczero_training.model.shared import Ffn, compute_swiglu_hidden_size
from lczero_training.model.utils import get_activation


class TestComputeSwigluHiddenSize:
    """Test the compute_swiglu_hidden_size helper function."""

    def test_basic_computation(self) -> None:
        """Test basic hidden size computation."""
        # To match 4d MLP (mlp_hidden=4096): (2/3)*4096 = 2730.67, ceil = 2731, round to 128 = 2816
        assert compute_swiglu_hidden_size(4096, multiple=128) == 2816

    def test_different_multiples(self) -> None:
        """Test with different rounding multiples."""
        mlp_hidden = 4096  # 4d for d=1024
        # (2/3) * 4096 = 2730.67, ceil = 2731
        # Round to 64: 2752
        assert compute_swiglu_hidden_size(mlp_hidden, multiple=64) == 2752
        # Round to 256: 2816
        assert compute_swiglu_hidden_size(mlp_hidden, multiple=256) == 2816

    def test_various_dimensions(self) -> None:
        """Test with various MLP hidden sizes."""
        # mlp_hidden=1024 (4d for d=256): (2/3)*1024 = 682.67, ceil = 683, round to 128 = 768
        assert compute_swiglu_hidden_size(1024, multiple=128) == 768
        # mlp_hidden=2048 (4d for d=512): (2/3)*2048 = 1365.33, ceil = 1366, round to 128 = 1408
        assert compute_swiglu_hidden_size(2048, multiple=128) == 1408


class TestSwigluFfnParamCount:
    """Test that SwiGLU FFN has approximately matching parameters to standard MLP."""

    def test_param_count_approximately_equal(self) -> None:
        """
        Compare SwiGLU vs standard MLP parameter counts.

        Standard MLP with dff=4d:
            params = 2 * d * 4d + (4d + d) biases = 8d^2 + 5d

        SwiGLU with dff=(8/3)d:
            params = 3 * d * (8/3)d + (2*(8/3)d + d) biases = 8d^2 + (19/3)d

        The kernel params are exactly equal (8d^2), bias difference is small.
        """
        d = 1024
        deepnorm_beta = 1.0
        rngs = nnx.Rngs(params=42)

        # Standard MLP with dff=4d
        mlp_dff = 4 * d
        mlp_ffn = Ffn(
            in_features=d,
            hidden_features=mlp_dff,
            hidden_activation=net_pb2.NetworkFormat.ACTIVATION_MISH,
            deepnorm_beta=deepnorm_beta,
            rngs=rngs,
        )

        # SwiGLU FFN with param-matched hidden size (matching 4d MLP)
        swiglu_dff = compute_swiglu_hidden_size(mlp_dff, multiple=128)
        swiglu_ffn = Ffn(
            in_features=d,
            hidden_features=swiglu_dff,
            hidden_activation=net_pb2.NetworkFormat.ACTIVATION_SWIGLU,
            deepnorm_beta=deepnorm_beta,
            rngs=rngs,
        )

        # Calculate expected param counts directly
        # MLP: linear1 (d -> 4d) + linear2 (4d -> d)
        # Kernels: d*4d + 4d*d = 8d^2
        # Biases: 4d + d = 5d
        mlp_kernel_params = 2 * d * mlp_dff  # d->4d and 4d->d
        mlp_bias_params = mlp_dff + d  # 4d + d
        mlp_params = mlp_kernel_params + mlp_bias_params

        # SwiGLU: gate_proj (d -> dff) + up_proj (d -> dff) + down_proj (dff -> d)
        # Kernels: d*dff + d*dff + dff*d = 3*d*dff
        # Biases: dff + dff + d
        swiglu_kernel_params = 3 * d * swiglu_dff
        swiglu_bias_params = 2 * swiglu_dff + d
        swiglu_params = swiglu_kernel_params + swiglu_bias_params

        # Verify by checking actual model attributes
        assert mlp_ffn.linear1.kernel.value.shape == (d, mlp_dff)
        assert mlp_ffn.linear2.kernel.value.shape == (mlp_dff, d)
        assert swiglu_ffn.gate_proj.kernel.value.shape == (d, swiglu_dff)
        assert swiglu_ffn.up_proj.kernel.value.shape == (d, swiglu_dff)
        assert swiglu_ffn.down_proj.kernel.value.shape == (swiglu_dff, d)

        # Compare kernel params (the dominant term)
        # MLP kernels = 8d^2 = 8*1024*1024 = 8,388,608
        # SwiGLU kernels with dff=2816: 3*1024*2816 = 8,650,752
        kernel_ratio = swiglu_kernel_params / mlp_kernel_params

        # Allow 5% tolerance for rounding differences in hidden size
        assert 0.95 <= kernel_ratio <= 1.10, (
            f"SwiGLU kernel params ({swiglu_kernel_params:,}) should be within 10% of "
            f"MLP kernel params ({mlp_kernel_params:,}), ratio: {kernel_ratio:.3f}"
        )

        # Total params ratio (including biases)
        total_ratio = swiglu_params / mlp_params
        assert 0.95 <= total_ratio <= 1.10, (
            f"SwiGLU total params ({swiglu_params:,}) should be within 10% of "
            f"MLP total params ({mlp_params:,}), ratio: {total_ratio:.3f}"
        )


class TestSwigluFfnForward:
    """Test SwiGLU FFN forward pass."""

    def test_swiglu_forward_shape(self) -> None:
        """Test that SwiGLU FFN produces correct output shape."""
        d = 256
        batch_size = 64
        deepnorm_beta = 1.0
        rngs = nnx.Rngs(params=42)

        ffn = Ffn(
            in_features=d,
            hidden_features=compute_swiglu_hidden_size(4 * d),
            hidden_activation=net_pb2.NetworkFormat.ACTIVATION_SWIGLU,
            deepnorm_beta=deepnorm_beta,
            rngs=rngs,
        )

        key = jax.random.key(0)
        x = jax.random.normal(key, (batch_size, d))
        output = ffn(x)

        assert output.shape == (batch_size, d)

    def test_swiglu_is_swiglu_flag(self) -> None:
        """Test that is_swiglu flag is set correctly."""
        d = 256
        deepnorm_beta = 1.0
        rngs = nnx.Rngs(params=42)

        swiglu_ffn = Ffn(
            in_features=d,
            hidden_features=compute_swiglu_hidden_size(4 * d),
            hidden_activation=net_pb2.NetworkFormat.ACTIVATION_SWIGLU,
            deepnorm_beta=deepnorm_beta,
            rngs=rngs,
        )
        assert swiglu_ffn.is_swiglu is True

        mish_ffn = Ffn(
            in_features=d,
            hidden_features=4 * d,
            hidden_activation=net_pb2.NetworkFormat.ACTIVATION_MISH,
            deepnorm_beta=deepnorm_beta,
            rngs=rngs,
        )
        assert mish_ffn.is_swiglu is False

    def test_swiglu_has_correct_projections(self) -> None:
        """Test that SwiGLU FFN has gate_proj, up_proj, down_proj."""
        d = 256
        dff = compute_swiglu_hidden_size(4 * d)
        deepnorm_beta = 1.0
        rngs = nnx.Rngs(params=42)

        ffn = Ffn(
            in_features=d,
            hidden_features=dff,
            hidden_activation=net_pb2.NetworkFormat.ACTIVATION_SWIGLU,
            deepnorm_beta=deepnorm_beta,
            rngs=rngs,
        )

        # Check SwiGLU has the right attributes
        assert hasattr(ffn, "gate_proj")
        assert hasattr(ffn, "up_proj")
        assert hasattr(ffn, "down_proj")
        assert not hasattr(ffn, "linear1")
        assert not hasattr(ffn, "linear2")

        # Check shapes
        assert ffn.gate_proj.kernel.value.shape == (d, dff)
        assert ffn.up_proj.kernel.value.shape == (d, dff)
        assert ffn.down_proj.kernel.value.shape == (dff, d)


class TestStandardMlpUnchanged:
    """Test that standard MLP activations still work."""

    def test_mish_activation_works(self) -> None:
        """Test MISH activation FFN works correctly."""
        d = 256
        dff = 4 * d
        deepnorm_beta = 1.0
        rngs = nnx.Rngs(params=42)

        ffn = Ffn(
            in_features=d,
            hidden_features=dff,
            hidden_activation=net_pb2.NetworkFormat.ACTIVATION_MISH,
            deepnorm_beta=deepnorm_beta,
            rngs=rngs,
        )

        key = jax.random.key(0)
        x = jax.random.normal(key, (64, d))
        output = ffn(x)

        assert output.shape == (64, d)
        assert hasattr(ffn, "linear1")
        assert hasattr(ffn, "linear2")
        assert not hasattr(ffn, "gate_proj")

    def test_swish_activation_works(self) -> None:
        """Test SWISH activation FFN works correctly."""
        d = 256
        dff = 4 * d
        deepnorm_beta = 1.0
        rngs = nnx.Rngs(params=42)

        ffn = Ffn(
            in_features=d,
            hidden_features=dff,
            hidden_activation=net_pb2.NetworkFormat.ACTIVATION_SWISH,
            deepnorm_beta=deepnorm_beta,
            rngs=rngs,
        )

        key = jax.random.key(0)
        x = jax.random.normal(key, (64, d))
        output = ffn(x)

        assert output.shape == (64, d)


class TestGetActivationSwigluError:
    """Test that get_activation raises error for SWIGLU."""

    def test_get_activation_raises_for_swiglu(self) -> None:
        """Test that get_activation raises ValueError for SWIGLU."""
        with pytest.raises(ValueError, match="ACTIVATION_SWIGLU is not a pointwise"):
            get_activation(net_pb2.NetworkFormat.ACTIVATION_SWIGLU)

    def test_get_activation_works_for_others(self) -> None:
        """Test that get_activation still works for other activations."""
        # These should all work without error
        activations = [
            net_pb2.NetworkFormat.ACTIVATION_MISH,
            net_pb2.NetworkFormat.ACTIVATION_RELU,
            net_pb2.NetworkFormat.ACTIVATION_SWISH,
            net_pb2.NetworkFormat.ACTIVATION_SELU,
        ]
        for activation in activations:
            fn = get_activation(activation)
            # Test the activation function works
            x = jnp.array([1.0, -1.0, 0.0])
            result = fn(x)
            assert result.shape == x.shape


class TestFullModelWithSwiglu:
    """Test full model instantiation with SwiGLU."""

    def test_model_runs_with_swiglu(self) -> None:
        """Test that the full model runs with SwiGLU activation."""
        from proto import model_config_pb2
        from proto.hlo_pb2 import XlaShapeProto

        from lczero_training.model.model import LczeroModel

        config = model_config_pb2.ModelConfig()
        config.defaults.compute_dtype = XlaShapeProto.BF16
        config.defaults.activation = net_pb2.NetworkFormat.ACTIVATION_MISH
        config.defaults.ffn_activation = net_pb2.NetworkFormat.ACTIVATION_SWIGLU

        # Use smaller dimensions for faster test
        d = 256
        config.embedding.dense_size = 128
        config.embedding.embedding_size = d
        config.embedding.dff = compute_swiglu_hidden_size(4 * d)

        config.encoder.num_blocks = 2
        config.encoder.d_model = d
        config.encoder.heads = 8
        config.encoder.dff = compute_swiglu_hidden_size(4 * d)

        config.encoder.smolgen.hidden_channels = 16
        config.encoder.smolgen.hidden_size = 64
        config.encoder.smolgen.gen_size = 64
        config.encoder.smolgen.activation = net_pb2.NetworkFormat.ACTIVATION_SWISH

        config.policy_head.embedding_size = d
        config.policy_head.d_model = d
        config.value_head.num_channels = 32
        config.movesleft_head.num_channels = 16

        rngs = nnx.Rngs(params=42)
        model = LczeroModel(config=config, rngs=rngs)

        # Run forward pass
        key = jax.random.key(0)
        random_input = jax.random.normal(key, (112, 8, 8))
        value, policy, movesleft = model(random_input)

        # Check output shapes
        assert value.shape == (3,)  # WDL values
        assert policy.shape == (1858,)  # Policy logits
        assert movesleft.shape == (1,)  # Moves left
