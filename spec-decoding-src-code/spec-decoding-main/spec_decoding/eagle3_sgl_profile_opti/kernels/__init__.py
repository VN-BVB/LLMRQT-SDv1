# SPDX-License-Identifier: Apache-2.0
"""Triton extend attention kernels (vendored from SGLang)."""

from .extend_attention import extend_attention_fwd

__all__ = ["extend_attention_fwd"]
