# SPDX-License-Identifier: Apache-2.0
"""Triton extend attention kernels (vendored third-party, Apache-2.0)."""

from .extend_attention import extend_attention_fwd

__all__ = ["extend_attention_fwd"]
