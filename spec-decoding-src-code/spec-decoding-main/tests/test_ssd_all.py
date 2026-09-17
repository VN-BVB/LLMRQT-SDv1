# SPDX-License-Identifier: Apache-2.0
"""Run all CPU-side spec_decoding.ssd unit tests (Phase 1–3)."""

from __future__ import annotations

import importlib
import unittest

_SSD_TEST_MODULES = (
    "tests.test_ssd_config",
    "tests.test_ssd_hidden",
    "tests.test_ssd_vocab",
    "tests.test_ssd_draft_chain",
    "tests.test_ssd_verify_accept",
    "tests.test_ssd_runner",
    "tests.test_ssd_fork",
    "tests.test_ssd_spec_cache",
)


def _suite_from_module(mod_name: str) -> unittest.TestSuite:
    mod = importlib.import_module(mod_name)
    return unittest.TestLoader().loadTestsFromModule(mod)


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern=None):
    suite = unittest.TestSuite()
    for name in _SSD_TEST_MODULES:
        suite.addTests(_suite_from_module(name))
    return suite


if __name__ == "__main__":
    suite = load_tests(unittest.TestLoader(), unittest.TestSuite())
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
