"""Three interchangeable attention kernels (TDD §4.3).

All three take ``q, k, v`` of shape ``[B, H, T, D]`` with KV heads already
expanded, and must return identical output within floating-point tolerance —
``test_all_impls_agree`` is the gate.

- ``manual_attention`` — the readable reference and the correctness oracle.
- ``sdpa_attention``   — dispatches to torch's fused kernels (including on MPS).
- ``flash_attention``  — our own tiled/online-softmax version, which never
  materialises the ``T x T`` score matrix.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``[B, n_kv_heads, T, D]`` to ``[B, n_kv_heads * n_rep, T, D]``.

    Query head ``h`` is paired with KV head ``h // n_rep``, so the repeats of a
    given KV head must land in *contiguous* output slots. Getting this ordering
    wrong (e.g. tiling instead of interleaving) silently mispairs every query
    head with the wrong group: shapes stay valid, the model still trains, it is
    just quietly worse. Hence ``test_repeat_kv_maps_query_head_to_its_group``.
    """
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, n_kv_heads, n_rep, seq, head_dim)
        .reshape(batch, n_kv_heads * n_rep, seq, head_dim)
    )


def causal_block_mask(q_len: int, k_len: int, offset: int, device: torch.device) -> torch.Tensor:
    """``[q_len, k_len]`` bool mask, ``True`` where attention is **disallowed**.

    Aligned bottom-right via ``offset``: query row *i* sits at absolute position
    ``offset + i`` and may see keys ``0 .. offset + i``. This matters whenever
    ``q_len != k_len`` (chunked prefill against a warm cache) — torch's
    ``is_causal=True`` aligns the triangle *top-left* instead, which would let
    early queries see the future and hide most of the cache from later ones.
    """
    q_pos = torch.arange(q_len, device=device) + offset
    k_pos = torch.arange(k_len, device=device)
    return k_pos[None, :] > q_pos[:, None]


def manual_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Textbook ``softmax(QKᵀ / √d) V``. Allocates the full ``[B,H,Tq,Tk]`` scores."""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    probs = F.softmax(scores, dim=-1)
    if dropout_p > 0.0 and training:
        probs = F.dropout(probs, p=dropout_p)
    return torch.matmul(probs, v)


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """torch's fused kernel. ``mask`` follows our convention (True = blocked)."""
    # sdpa's bool convention is the inverse of ours: True means *participate*.
    attn_mask = None if mask is None else ~mask
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        is_causal=is_causal,
        dropout_p=dropout_p if training else 0.0,
    )


def _flash_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    offset: int = 0,
    block_q: int = 64,
    block_k: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tiled forward pass. Returns ``(out, logsumexp)``.

    The log-sum-exp per query row is the *only* extra state the backward pass
    needs. It is O(T) — one scalar per query — which is what lets backward
    recompute the score tiles instead of the forward saving them (TDR-021).

    Walks query tiles on the outside and key tiles on the inside, carrying a
    running max ``m`` and running denominator ``l`` per query row. When a key
    tile raises the max, the accumulator is rescaled by ``exp(m_old - m_new)``
    before the new contribution is added — algebraically identical to a single
    global softmax, but only one ``[block_q, block_k]`` tile is ever resident,
    so peak memory is O(T) rather than O(T²).

    Two edge cases that produce NaN if unhandled:

    - A key tile entirely beyond the causal frontier leaves the row max at
      ``-inf``, and ``exp(-inf - -inf)`` is NaN. Rows are clamped to a finite
      max before the exponential.
    - Under causal masking, key tiles past the last query position contribute
      nothing at all, so the loop breaks out rather than computing them — this
      is where the tiled form actually saves work, not just memory.
    """
    *_, q_len, head_dim = q.shape
    k_len = k.shape[-2]
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(q)
    lse = torch.empty((*q.shape[:-1], 1), device=q.device, dtype=q.dtype)

    for i in range(0, q_len, block_q):
        q_tile = q[:, :, i : i + block_q]
        tile_q = q_tile.shape[2]
        q_pos = torch.arange(i, i + tile_q, device=q.device) + offset

        shape = (*q.shape[:2], tile_q, 1)
        running_max = torch.full(shape, float("-inf"), device=q.device, dtype=q.dtype)
        running_sum = torch.zeros(shape, device=q.device, dtype=q.dtype)
        acc = torch.zeros_like(q_tile)

        for j in range(0, k_len, block_k):
            # Every later key tile is fully masked too, so stop rather than
            # burn work on tiles that contribute nothing.
            if causal and j > int(q_pos[-1].item()):
                break

            k_tile = k[:, :, j : j + block_k]
            v_tile = v[:, :, j : j + block_k]
            scores = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * scale

            if causal:
                k_pos = torch.arange(j, j + k_tile.shape[2], device=q.device)
                scores = scores.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))

            new_max = torch.maximum(running_max, scores.amax(dim=-1, keepdim=True))
            # Rows still fully masked have max -inf; pin them to 0 so the
            # exponentials below evaluate to exactly 0 instead of NaN. Safe
            # because such rows also have running_sum == 0 and acc == 0, so the
            # choice of origin cannot affect the final quotient.
            new_max = torch.where(torch.isneginf(new_max), torch.zeros_like(new_max), new_max)

            rescale = torch.exp(running_max - new_max)
            probs = torch.exp(scores - new_max)
            running_sum = running_sum * rescale + probs.sum(dim=-1, keepdim=True)
            acc = acc * rescale + torch.matmul(probs, v_tile)
            running_max = new_max

        # Causal attention always leaves a row at least its own key, so the
        # denominator is positive; clamp only to keep a degenerate all-masked
        # call from producing NaN instead of zeros.
        denom = running_sum.clamp_min(torch.finfo(q.dtype).tiny)
        out[:, :, i : i + tile_q] = acc / denom
        # log-sum-exp in the original (unshifted) score space, so backward can
        # rebuild the exact probabilities without knowing the tile order.
        lse[:, :, i : i + tile_q] = running_max + torch.log(denom)

    return out, lse


def _flash_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    causal: bool = True,
    offset: int = 0,
    block_q: int = 64,
    block_k: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute score tiles and accumulate ``dq``, ``dk``, ``dv`` (TDR-021).

    With ``L`` (the saved log-sum-exp) the probabilities are recoverable exactly:
    ``p_ij = exp(s_ij - L_i)``, no running max needed. Differentiating
    ``o_i = Σ_j p_ij v_j`` then gives

        D_i   = Σ_j p_ij (do_i · v_j) = do_i · o_i
        dv_j += Σ_i p_ij do_i
        dp_ij = do_i · v_j
        ds_ij = p_ij (dp_ij − D_i)
        dq_i += scale Σ_j ds_ij k_j
        dk_j += scale Σ_i ds_ij q_i

    ``D_i`` collapses to a row-wise dot product of ``dout`` and ``out``, which is
    why the forward only has to keep ``out`` and ``L`` — the ``T × T`` matrices
    ``p``, ``dp`` and ``ds`` exist one tile at a time and are discarded.
    """
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    q_len, k_len = q.shape[-2], k.shape[-2]

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    row_dot = (dout * out).sum(dim=-1, keepdim=True)  # D_i

    for i in range(0, q_len, block_q):
        span_q = slice(i, i + block_q)
        q_tile = q[:, :, span_q]
        dout_tile = dout[:, :, span_q]
        lse_tile = lse[:, :, span_q]
        dot_tile = row_dot[:, :, span_q]
        tile_q = q_tile.shape[2]
        q_pos = torch.arange(i, i + tile_q, device=q.device) + offset
        dq_tile = torch.zeros_like(q_tile)

        for j in range(0, k_len, block_k):
            if causal and j > int(q_pos[-1].item()):
                break

            span_k = slice(j, j + block_k)
            k_tile = k[:, :, span_k]
            v_tile = v[:, :, span_k]

            scores = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * scale
            if causal:
                k_pos = torch.arange(j, j + k_tile.shape[2], device=q.device)
                scores = scores.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))

            probs = torch.exp(scores - lse_tile)
            dv[:, :, span_k] += torch.matmul(probs.transpose(-2, -1), dout_tile)

            dprobs = torch.matmul(dout_tile, v_tile.transpose(-2, -1))
            dscores = probs * (dprobs - dot_tile)

            dq_tile += torch.matmul(dscores, k_tile) * scale
            dk[:, :, span_k] += torch.matmul(dscores.transpose(-2, -1), q_tile) * scale

        dq[:, :, span_q] = dq_tile

    return dq, dk, dv


class FlashAttentionFn(torch.autograd.Function):
    """Autograd wrapper that keeps the O(T) memory claim true in *training*.

    Without this, the tiled kernel is plain PyTorch ops, so autograd saves every
    tile's intermediates for backward — and the tiles sum back to O(T²). Measured
    before this change: the tiled path saved *more* than the naive one (1.30×
    at T=512) and both grew quadratically, so the memory advantage existed only
    under ``no_grad``.

    Saving ``(q, k, v, out, lse)`` and recomputing the score tiles in backward
    trades a second pass over the tiles for O(T·d) saved state instead of O(T²).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        offset: int,
        block_q: int,
        block_k: int,
    ) -> torch.Tensor:
        out, lse = _flash_forward(
            q, k, v, causal=causal, offset=offset, block_q=block_q, block_k=block_k
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal, ctx.offset = causal, offset
        ctx.block_q, ctx.block_k = block_q, block_k
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[Any, ...]:  # type: ignore[override]
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_backward(
            dout.contiguous(),
            q,
            k,
            v,
            out,
            lse,
            causal=ctx.causal,
            offset=ctx.offset,
            block_q=ctx.block_q,
            block_k=ctx.block_k,
        )
        return dq, dk, dv, None, None, None, None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    offset: int = 0,
    block_q: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """Tiled attention with online softmax and a recomputing backward (TDR-007, TDR-021)."""
    out: torch.Tensor = FlashAttentionFn.apply(q, k, v, causal, offset, block_q, block_k)
    return out
