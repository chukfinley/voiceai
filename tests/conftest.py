"""pytest config — register the `slow` marker so we can skip GPU/network tests."""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: tests that download models or need GPU (deselect with -m 'not slow')"
    )
