"""Shared pytest configuration and markers."""


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU (run on H100)")
