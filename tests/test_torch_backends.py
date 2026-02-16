"""Tests for PyTorch backend configuration."""

import torch

from luckylab.utils.torch import configure_torch_backends


class TestConfigureTorchBackends:
    """Tests for configure_torch_backends."""

    def test_runs_without_error(self):
        configure_torch_backends()

    def test_allow_tf32_parameter(self):
        configure_torch_backends(allow_tf32=True)
        configure_torch_backends(allow_tf32=False)

    def test_deterministic_sets_cudnn(self):
        configure_torch_backends(deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_non_deterministic_sets_cudnn(self):
        configure_torch_backends(deterministic=False)
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
