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
    """Tiled attention with online softmax (TDR-007).

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
        out[:, :, i : i + tile_q] = acc / running_sum.clamp_min(torch.finfo(q.dtype).tiny)

    return out
