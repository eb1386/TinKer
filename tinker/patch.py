"""model patcher"""

import torch.nn as nn

from tinker.modules.attention import TinKerAttention, _find_attention_projections
from tinker.modules.feed_forward import TinKerSwiGLU, _find_swiglu_weights
from tinker.modules.normalization import TinKerRMSNorm


def _is_rmsnorm(module: nn.Module) -> bool:
    """detect rmsnorm"""
    if not hasattr(module, "weight"):
        return False
    if not isinstance(module.weight, nn.Parameter):
        return False
    if module.weight.dim() != 1:
        return False
    if not hasattr(module, "eps") and not hasattr(module, "variance_epsilon"):
        return False
    if hasattr(module, "bias") and module.bias is not None:
        return False
    return True


def _is_swiglu(module: nn.Module) -> bool:
    """detect swiglu"""
    try:
        _find_swiglu_weights(module)
        return True
    except (ValueError, AttributeError):
        return False


def _is_gqa_attention(module: nn.Module) -> bool:
    """detect gqa attention"""
    try:
        _find_attention_projections(module)
        return True
    except (ValueError, AttributeError):
        return False


def patch_model(
    model: nn.Module,
    replace_attention: bool = True,
    replace_ffn: bool = True,
    replace_norm: bool = True,
    rope_theta: float = 10000.0,
    max_seq_len: int = 2048,
    verbose: bool = False,
) -> nn.Module:
    """patch model modules"""
    counts = {"attention": 0, "ffn": 0, "norm": 0}

    replacements: list[tuple[nn.Module, str, nn.Module]] = []

    for parent_name, parent_module in model.named_modules():
        for name, child in parent_module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            if replace_norm and _is_rmsnorm(child):
                try:
                    replacement = TinKerRMSNorm.from_module(child)
                    replacement = replacement.to(
                        device=child.weight.device, dtype=child.weight.dtype
                    )
                    replacements.append((parent_module, name, replacement))
                    counts["norm"] += 1
                    if verbose:
                        print(f"  [tinker] rmsnorm -> {full_name}")
                except Exception as e:
                    if verbose:
                        print(f"  [tinker] skipped rmsnorm {full_name}: {e}")

            elif replace_ffn and _is_swiglu(child):
                try:
                    replacement = TinKerSwiGLU.from_module(child)
                    gate_w = next(
                        getattr(child, n).weight
                        for n in ["gate_proj", "w1", "gate"]
                        if hasattr(child, n)
                    )
                    replacement = replacement.to(
                        device=gate_w.device, dtype=gate_w.dtype
                    )
                    replacements.append((parent_module, name, replacement))
                    counts["ffn"] += 1
                    if verbose:
                        print(f"  [tinker] swiglu -> {full_name}")
                except Exception as e:
                    if verbose:
                        print(f"  [tinker] skipped swiglu {full_name}: {e}")

            elif replace_attention and _is_gqa_attention(child):
                try:
                    replacement = TinKerAttention.from_module(
                        child, rope_theta=rope_theta, max_seq_len=max_seq_len
                    )
                    q_proj = next(
                        getattr(child, n)
                        for n in ["q_proj", "query", "wq", "q"]
                        if hasattr(child, n)
                    )
                    replacement = replacement.to(
                        device=q_proj.weight.device, dtype=q_proj.weight.dtype
                    )
                    replacements.append((parent_module, name, replacement))
                    counts["attention"] += 1
                    if verbose:
                        print(f"  [tinker] attention -> {full_name}")
                except Exception as e:
                    if verbose:
                        print(f"  [tinker] skipped attention {full_name}: {e}")

    for parent, attr_name, new_module in replacements:
        setattr(parent, attr_name, new_module)

    if verbose:
        total = sum(counts.values())
        print(f"\n  [tinker] patched {total} modules: "
              f"{counts['attention']} attention, {counts['ffn']} ffn, {counts['norm']} norm")

    return model
