# SPDX-License-Identifier: Apache-2.0
"""Run all CPU-side EAGLE2 unit tests (no transformers / GPU required)."""

from __future__ import annotations

import importlib
import unittest

_EAGLE2_TEST_MODULES = (
    "tests.test_eagle2_tree_verify_mask",
    "tests.test_eagle2_full_tree",
    "tests.test_eagle2_cumulative_prune",
    "tests.test_eagle2_verify_attn",
)


def _suite_from_module(mod_name: str) -> unittest.TestSuite:
    mod = importlib.import_module(mod_name)
    suite = unittest.TestSuite()
    for name in sorted(dir(mod)):
        if name.startswith("test_"):
            fn = getattr(mod, name)
            suite.addTest(unittest.FunctionTestCase(fn))
    return suite


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern=None):
    suite = unittest.TestSuite()
    for name in _EAGLE2_TEST_MODULES:
        suite.addTests(_suite_from_module(name))
    return suite


if __name__ == "__main__":
    suite = load_tests(unittest.TestLoader(), unittest.TestSuite())
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
