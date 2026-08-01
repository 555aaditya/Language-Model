"""Post-training weight quantization (TDD §7.1, TDR-012).

**Weight-only, symmetric, per-output-channel.** Weights are stored at reduced
precision and expanded back to the activation dtype inside `forward`, so the
matmul itself still runs in float.

That means this buys **memory, not speed** — and on some hardware it is
measurably *slower* than fp32 because of the dequantization step. TDD §7.1
already carries that caveat; the `benchmark/` layer measures both per device
rather than assuming a winner. Real int8 throughput needs fused integer
kernels, which is a different piece of work from this one.

Per-output-channel scales matter more than they look. A single per-tensor scale
is set by the largest weight anywhere in the matrix, so one outlier row flattens
every other row toward zero — `test_int8_scales_are_per_output_channel_not_per_tensor`
pins that.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

VALID_SCHEMES = ("int_weight", "fp16", "bf16", "none")
VALID_BITS = (4, 8)

# Symmetric signed range, one level held back so the range stays centred:
# int8 uses [-127, 127] rather than [-128, 127], int4 uses [-7, 7].
_QMAX = {8: 127, 4: 7}


def _scales(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-output-channel scale, guarded against an all-zero row."""
    amax = weight.abs().amax(dim=1, keepdim=True)
    # A row of exact zeros would give scale 0 and produce nan on divide. Any
    # positive scale reproduces that row exactly, so 1.0 is as good as anything.
    return torch.where(amax == 0, torch.ones_like(amax), amax / _QMAX[bits])


def quantize_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[out, in]`` float → ``(int8 codes, per-channel scales)``."""
    scales = _scales(weight, 8)
    codes = torch.round(weight / scales).clamp(-127, 127).to(torch.int8)
    return codes, scales


def dequantize_int8(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return codes.to(scales.dtype) * scales


def quantize_int4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``[out, in]`` float → ``(uint8 packed [out, ceil(in/2)], scales, in_features)``.

    Two 4-bit codes share one byte: low nibble first, high nibble second. Codes
    are biased by +8 so the signed range ``[-7, 7]`` fits the unsigned nibble.
    An odd ``in`` is zero-padded to the next even width, so the original width
    has to be carried alongside in order to unpack.
    """
    out_features, in_features = weight.shape
    scales = _scales(weight, 4)
    codes = torch.round(weight / scales).clamp(-7, 7).to(torch.int16) + 8

    if in_features % 2:
        codes = F.pad(codes, (0, 1), value=8)  # 8 == biased zero

    low, high = codes[:, 0::2], codes[:, 1::2]
    packed = (low | (high << 4)).to(torch.uint8)
    return packed, scales, in_features


def dequantize_int4(packed: torch.Tensor, scales: torch.Tensor, in_features: int) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int16) - 8
    high = (packed >> 4).to(torch.int16) - 8

    interleaved = torch.stack((low, high), dim=2).reshape(packed.shape[0], -1)
    return interleaved[:, :in_features].to(scales.dtype) * scales


class QuantizedLinear(nn.Module):
    """A drop-in ``nn.Linear`` holding reduced-precision weights.

    Weights live in buffers, not parameters: they carry no gradient and must not
    reappear in an optimizer. Quantization here is a post-training step, so a
    quantised layer is not trainable, and making that structural is safer than
    documenting it.
    """

    # Declared so the type checker sees Tensors; `register_buffer` alone leaves
    # these typed as `Tensor | Module`.
    codes: torch.Tensor
    scales: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        *,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if bits not in VALID_BITS:
            raise ValueError(f"bits must be one of {VALID_BITS}, got {bits}")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.register_buffer("codes", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("scales", torch.empty(0))
        # Bias stays float: one value per output channel, so quantising it saves
        # nothing measurable and only adds error.
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, bits: int = 8) -> QuantizedLinear:
        layer = cls(linear.in_features, linear.out_features, bits, bias=linear.bias)
        weight = linear.weight.detach().float()
        if bits == 8:
            codes, scales = quantize_int8(weight)
            layer.codes = codes.view(torch.uint8)
        else:
            packed, scales, _ = quantize_int4(weight)
            layer.codes = packed
        layer.scales = scales
        return layer

    def dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if self.bits == 8:
            weight = dequantize_int8(self.codes.view(torch.int8), self.scales)
        else:
            weight = dequantize_int4(self.codes, self.scales, self.in_features)
        return weight.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.dequantized_weight(x.dtype), bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bits={self.bits}"


def model_nbytes(module: nn.Module) -> int:
    """Total bytes held by parameters *and* buffers.

    Buffers matter: quantised weights are buffers, so a parameters-only count
    would report a quantised model as almost empty and make any saving look
    spectacular and fake.
    """
    seen: set[int] = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor is None or id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor.numel() * tensor.element_size()
    return total


def _tied_tensors(model: nn.Module) -> set[int]:
    """Ids of tensors reachable under more than one name."""
    counts: dict[int, int] = {}
    for _, tensor in model.named_parameters():
        counts[id(tensor)] = counts.get(id(tensor), 0) + 1
    for _, tensor in model.named_parameters(remove_duplicate=False):
        counts[id(tensor)] = counts.get(id(tensor), 0) + 1
    return {i for i, n in counts.items() if n > 2}


def quantize(
    model: nn.Module,
    *,
    bits: int = 8,
    scheme: str = "int_weight",
    skip: tuple[str, ...] = ("lm_head",),
    inplace: bool = False,
) -> nn.Module:
    """Quantize ``model`` (docs/BUILD_ORDER.md contract). Returns the model.

    ``skip`` defaults to the LM head because it is usually **tied** to the token
    embedding. Replacing it with a quantised copy silently severs that tie: the
    head becomes int8 while the embedding stays fp32, so two matrices that are
    meant to be one tensor drift apart the moment anything touches them. It also
    saves nothing, since the embedding keeps the float copy alive regardless.
    """
    if scheme not in VALID_SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {VALID_SCHEMES}")
    if scheme == "int_weight" and bits not in VALID_BITS:
        raise ValueError(f"bits must be one of {VALID_BITS}, got {bits}")

    target = model if inplace else copy.deepcopy(model)

    if scheme == "none":
        return target
    if scheme in ("fp16", "bf16"):
        return target.to(torch.float16 if scheme == "fp16" else torch.bfloat16)

    tied = _tied_tensors(target)
    for parent_name, parent in list(target.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(full == s or full.endswith(f".{s}") for s in skip):
                continue
            if id(child.weight) in tied:
                continue
            setattr(parent, child_name, QuantizedLinear.from_linear(child, bits=bits))

    return target
