"""``CausalLM`` — the full decoder-only language model.

Contract (docs/BUILD_ORDER.md):

    CausalLM.forward(input_ids, *, kv_cache=None, use_cache=False) -> logits [B, T, V]
    CausalLM.from_config(cfg: dict) -> CausalLM

**forward returns logits, never a loss.** Cross-entropy lives in the trainer.
Computing a loss during generation is wasted work, and keeping the model pure
means the inference path has exactly one thing to reason about.

**The KV cache is caller-owned.** Because the contract fixes the return type to
logits, there is nowhere to hand a freshly created cache back. So the caller
allocates one with ``model.new_cache()`` and passes the same list on every
step; the per-layer ``KVCache`` objects are mutated in place. Passing
``use_cache=True`` without a cache raises rather than silently discarding it —
a discarded cache turns O(T) generation back into O(T²) while still producing
correct text, which is exactly the kind of bug that never gets noticed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from attention import KVCache
from model.block import TransformerBlock
from model.norm import RMSNorm


class CausalLM(nn.Module):
    """Embedding → N pre-norm blocks → RMSNorm → LM head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        d_ff: int | None = None,
        *,
        max_seq_len: int = 1024,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        impl: str = "sdpa",
        bias: bool = False,
        norm_eps: float = 1e-6,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        d_ff = 4 * d_model if d_ff is None else d_ff

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        # Kept as a plain list as well as a ModuleList: iterating the ModuleList
        # yields `Module`, which loses the block's attribute types.
        blocks = [
            TransformerBlock(
                d_model,
                n_heads,
                n_kv_heads,
                d_ff,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                dropout=dropout,
                impl=impl,
                bias=bias,
                norm_eps=norm_eps,
            )
            for _ in range(n_layers)
        ]
        self.blocks = blocks
        self.layers = nn.ModuleList(blocks)
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)
        # Residual output projections are scaled down by 1/sqrt(2 * n_layers):
        # every block adds two contributions to the residual stream, so without
        # this the stream's variance grows linearly with depth.
        residual_std = 0.02 / (2 * n_layers) ** 0.5
        for block in blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=residual_std)

        # Tie *after* init, or the head's initialisation would overwrite the
        # embedding's. `is`-identity matters: a copy would silently train two
        # separate matrices that were only equal at step 0.
        self.tie_weights = tie_weights
        if tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> CausalLM:
        """Build from a loaded YAML config's ``model`` / ``attention`` blocks."""
        m = cfg["model"]
        a = cfg.get("attention", {})
        return cls(
            vocab_size=int(m["vocab_size"]),
            d_model=int(m["d_model"]),
            n_layers=int(m["n_layers"]),
            n_heads=int(m["n_heads"]),
            n_kv_heads=int(m.get("n_kv_heads", m["n_heads"])),
            d_ff=int(m.get("d_ff", 4 * int(m["d_model"]))),
            max_seq_len=int(m.get("max_seq_len", 1024)),
            rope_theta=float(m.get("rope_theta", 10000.0)),
            dropout=float(m.get("dropout", 0.0)),
            impl=str(a.get("impl", "sdpa")),
            tie_weights=bool(m.get("tie_weights", True)),
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def new_cache(self) -> list[KVCache]:
        """One empty ``KVCache`` per layer, to be passed back on every step."""
        return [KVCache() for _ in range(self.n_layers)]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        kv_cache: list[KVCache] | None = None,
        use_cache: bool = False,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if use_cache and kv_cache is None:
            raise ValueError(
                "use_cache=True needs a caller-owned cache; call model.new_cache() "
                "once and pass the same list on every step"
            )
        if kv_cache is not None and len(kv_cache) != self.n_layers:
            raise ValueError(
                f"cache has {len(kv_cache)} entries but the model has {self.n_layers} layers"
            )

        offset = len(kv_cache[0]) if kv_cache else 0
        seq_len = input_ids.shape[1]
        if offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"context length {offset + seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        x = self.dropout(self.embed_tokens(input_ids))
        for i, block in enumerate(self.blocks):
            # Each layer keeps its own cache; sharing one across layers still
            # runs and still generates plausible text, so it is worth being
            # explicit that the indexing is per-layer.
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_cache = block(
                x,
                kv_cache=layer_cache,
                use_cache=use_cache,
                key_padding_mask=key_padding_mask,
            )
            if use_cache and kv_cache is not None and new_cache is not None:
                kv_cache[i] = new_cache

        logits: torch.Tensor = self.lm_head(self.norm(x))
        return logits

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def num_parameters(self, *, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
