"""tinker — inference optimization for small language models"""

from tinker.modules.attention import TinKerAttention
from tinker.modules.feed_forward import TinKerSwiGLU
from tinker.modules.normalization import TinKerRMSNorm
from tinker.patch import patch_model
from tinker.optimize import optimize, benchmark_model

__all__ = [
    "TinKerAttention",
    "TinKerSwiGLU",
    "TinKerRMSNorm",
    "patch_model",
    "optimize",
    "benchmark_model",
]

__version__ = "0.1.0"
