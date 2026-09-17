# SPDX-License-Identifier: Apache-2.0
"""Run all CPU-side EAGLE3 unit tests (Phase 0–2; no GPU required for most)."""

from __future__ import annotations

import importlib
import unittest

_EAGLE3_TEST_MODULES = (
    "tests.test_eagle3_config",
    "tests.test_eagle3_hidden",
    "tests.test_eagle3_vocab",
    "tests.test_eagle3_draft_chain",
    "tests.test_eagle3_verify_accept",
    "tests.test_eagle3_runner",
)


def _suite_from_module(mod_name: str) -> unittest.TestSuite:
    mod = importlib.import_module(mod_name)
    return unittest.TestLoader().loadTestsFromModule(mod)


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern=None):
    suite = unittest.TestSuite()
    for name in _EAGLE3_TEST_MODULES:
        suite.addTests(_suite_from_module(name))
    return suite


if __name__ == "__main__":
    suite = load_tests(unittest.TestLoader(), unittest.TestSuite())
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
