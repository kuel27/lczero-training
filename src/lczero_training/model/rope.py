# src/lczero_training/model/rope.py
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


def _rotate_pairs(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """
    x: (..., D) with D even
    cos/sin: broadcastable to (..., D/2)
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE requires an even last-dim, got {x.shape[-1]}")
    x_f = x.astype(jnp.float32)
    x2 = x_f.reshape(*x_f.shape[:-1], x_f.shape[-1] // 2, 2)
    x1 = x2[..., 0]
    x2v = x2[..., 1]
    y1 = x1 * cos - x2v * sin
    y2 = x1 * sin + x2v * cos
    return jnp.stack([y1, y2], axis=-1).reshape(x_f.shape)


def _build_axial_cos_sin_from_coords(
    *,
    xs: jax.Array,  # (N,)
    ys: jax.Array,  # (N,)
    dim_each: int,
    theta: float,
    dtype=jnp.float32,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    dim_each: rotary dims per axis (row-axis and col-axis), must be even.
    Returns cos/sin for x and y:
      cos_x, sin_x, cos_y, sin_y all shaped (N, dim_each/2)
    """
    if dim_each % 2 != 0:
        raise ValueError(f"dim_each must be even, got {dim_each}")
    if xs.ndim != 1 or ys.ndim != 1 or xs.shape != ys.shape:
        raise ValueError(f"xs/ys must be 1D and same shape, got {xs.shape=} {ys.shape=}")

    xs = xs.astype(jnp.float32)
    ys = ys.astype(jnp.float32)
    n = xs.shape[0]

    # Standard RoPE inv_freq pattern (pairs => step 2).
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim_each, 2, dtype=jnp.float32) / dim_each))  # (dim_each/2,)

    freqs_x = xs[:, None] * inv_freq[None, :]  # (N, dim_each/2)
    freqs_y = ys[:, None] * inv_freq[None, :]  # (N, dim_each/2)

    # Keep trig tables in float32 for stability; only cast at use-time if needed.
    cos_x = jnp.cos(freqs_x).astype(dtype)
    sin_x = jnp.sin(freqs_x).astype(dtype)
    cos_y = jnp.cos(freqs_y).astype(dtype)
    sin_y = jnp.sin(freqs_y).astype(dtype)

    if cos_x.shape != (n, dim_each // 2):
        raise RuntimeError(f"Unexpected trig table shape: {cos_x.shape} (expected {(n, dim_each // 2)})")
    return cos_x, sin_x, cos_y, sin_y


class RoPECache(nnx.Variable):
    """Non-trainable cached state (not differentiated by default)."""
    pass


class Axial2DRoPE(nnx.Module):
    """
    Cached (precomputed) axial 2D RoPE for an HxW grid.

    Applies RoPE to the first `rotary_dim` channels of Q/K (dim must be multiple of 4 after trimming).
    Works on batched MHA tensors shaped: (B, S, Heads, Dh) with Dh last.

    Token layout:
      - By default, assumes row-major board flattening.
      - You can supply `token_coords` (shape (H*W, 2) as (y, x)) to match your exact board token order.
      - You can set `board_start` in __call__ to place the board anywhere in the sequence
        (e.g., board_start=1 for a leading CLS token).

    If the sequence doesn't contain the full board, RoPE is applied to the available board tokens only.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        height: int = 8,
        width: int = 8,
        theta: float = 100.0,
        rotary_dim: int | None = None,
        token_coords: jax.Array | None = None,  # (H*W, 2) = (y, x)
        cache_dtype=jnp.float32,  # keep cache in float32 for stability
    ):
        self.height = int(height)
        self.width = int(width)
        self.theta = float(theta)
        self.head_dim = int(head_dim)

        if rotary_dim is None:
            rotary_dim = head_dim
        rotary_dim = int(rotary_dim)

        # Need: pairs + split into x/y halves => multiple of 4
        rotary_dim = max(0, min(rotary_dim, self.head_dim))
        rotary_dim -= rotary_dim % 4

        self.rotary_dim = rotary_dim
        self.dim_each = rotary_dim // 2  # per axis
        self.n_board = self.height * self.width

        # Establish token coordinate order (y, x) for the board tokens.
        if token_coords is None:
            ys = jnp.repeat(jnp.arange(self.height, dtype=jnp.float32), self.width)  # (H*W,)
            xs = jnp.tile(jnp.arange(self.width, dtype=jnp.float32), self.height)   # (H*W,)
        else:
            coords = jnp.asarray(token_coords, dtype=jnp.float32)
            if coords.shape != (self.n_board, 2):
                raise ValueError(f"token_coords must have shape {(self.n_board, 2)}, got {coords.shape}")
            ys = coords[:, 0]
            xs = coords[:, 1]

        if self.rotary_dim == 0:
            # Keep consistent attributes as non-trainable cache variables.
            z = jnp.zeros((0, 0), dtype=cache_dtype)
            self.cos_x = RoPECache(z)
            self.sin_x = RoPECache(z)
            self.cos_y = RoPECache(z)
            self.sin_y = RoPECache(z)
            return

        cos_x, sin_x, cos_y, sin_y = _build_axial_cos_sin_from_coords(
            xs=xs,
            ys=ys,
            dim_each=self.dim_each,
            theta=self.theta,
            dtype=cache_dtype,
        )

        self.cos_x = RoPECache(cos_x)
        self.sin_x = RoPECache(sin_x)
        self.cos_y = RoPECache(cos_y)
        self.sin_y = RoPECache(sin_y)

    def __call__(self, q: jax.Array, k: jax.Array, *, board_start: int = 0) -> tuple[jax.Array, jax.Array]:
        """
        q,k: (B, S, Heads, Dh)
        board_start: index of the first board token in the sequence.
        """
        if self.rotary_dim == 0:
            return q, k

        if q.ndim != 4:
            raise ValueError(f"Expected q shape (B,S,H,D), got {q.shape}")
        if k.shape != q.shape:
            raise ValueError(f"q and k must have same shape, got {q.shape} vs {k.shape}")

        B, S, H, Dh = q.shape
        if Dh != self.head_dim:
            raise ValueError(f"Expected head_dim={self.head_dim}, got {Dh}")

        board_start = int(board_start)
        if board_start < 0 or board_start > S:
            raise ValueError(f"board_start must be in [0, S], got {board_start} with S={S}")

        # Apply to as many board tokens as are available in the sequence.
        n_apply = min(self.n_board, S - board_start)
        if n_apply == 0:
            return q, k

        # Split: prefix | board_slice | suffix
        q_prefix = q[:, :board_start, :, :] if board_start > 0 else q[:, :0, :, :]
        k_prefix = k[:, :board_start, :, :] if board_start > 0 else k[:, :0, :, :]

        q_board = q[:, board_start : board_start + n_apply, :, :]
        k_board = k[:, board_start : board_start + n_apply, :, :]

        q_suffix = q[:, board_start + n_apply :, :, :]
        k_suffix = k[:, board_start + n_apply :, :, :]

        # Slice rotary dims and passthrough dims
        rot_q, pass_q = q_board[..., : self.rotary_dim], q_board[..., self.rotary_dim :]
        rot_k, pass_k = k_board[..., : self.rotary_dim], k_board[..., self.rotary_dim :]

        # Split x/y halves
        qx, qy = rot_q[..., : self.dim_each], rot_q[..., self.dim_each :]
        kx, ky = rot_k[..., : self.dim_each], rot_k[..., self.dim_each :]

        # Broadcast cos/sin to (B, n_apply, Heads, dim_each/2)
        cos_x = self.cos_x.value[None, :n_apply, None, :]
        sin_x = self.sin_x.value[None, :n_apply, None, :]
        cos_y = self.cos_y.value[None, :n_apply, None, :]
        sin_y = self.sin_y.value[None, :n_apply, None, :]

        qx_out = _rotate_pairs(qx, cos_x, sin_x).astype(q.dtype)
        kx_out = _rotate_pairs(kx, cos_x, sin_x).astype(k.dtype)
        qy_out = _rotate_pairs(qy, cos_y, sin_y).astype(q.dtype)
        ky_out = _rotate_pairs(ky, cos_y, sin_y).astype(k.dtype)

        q_board_out = jnp.concatenate([qx_out, qy_out, pass_q], axis=-1)
        k_board_out = jnp.concatenate([kx_out, ky_out, pass_k], axis=-1)

        # Re-attach: prefix | rotated_board | suffix
        q_out = jnp.concatenate([q_prefix, q_board_out, q_suffix], axis=1)
        k_out = jnp.concatenate([k_prefix, k_board_out, k_suffix], axis=1)
        return q_out, k_out
