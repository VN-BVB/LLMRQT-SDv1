#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run real-weight EAGLE3 on Qwen3-1.7B with 8 GB-friendly defaults.

The target and drafter must remain a matching pair.  This entry point does not
fall back to a Llama model when Qwen loading fails, which avoids accidentally
downloading or loading an 8B target on a small GPU.

Example:
    PYTHONPATH=. python3 -m \
        spec_decoding.eagle3_sgl.example.run_eagle3_sgl_qwen3
"""

from __future__ import annotations

from .run_eagle3_sgl_llama31 import main as run_eagle3


def main() -> None:
    run_eagle3(
        default_target="Qwen/Qwen3-1.7B",
        default_draft="AngelSlim/Qwen3-1.7B_eagle3",
        fallback_target=None,
        description="Real EAGLE3 on Qwen3-1.7B (8 GB-friendly defaults)",
        default_max_new_tokens=32,
        default_topk=1,
        default_num_steps=1,
        default_max_tree_nodes=1,
        default_verify_mode="full_model_tree",
        # 4060 实测最优默认：top-1 短链触发单 target-forward 快路径；参数仍可在 CLI 覆盖。
    )


if __name__ == "__main__":
    main()
