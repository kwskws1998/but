import pytest

from utils.runtime_compat import validate_cuda_runtime


class FakeCuda:
    def __init__(self, available, capability=(0, 0), architectures=None):
        self.available = available
        self.capability = capability
        self.architectures = architectures or []

    def is_available(self):
        return self.available

    def get_device_capability(self):
        return self.capability

    def get_arch_list(self):
        return self.architectures


class FakeTorch:
    def __init__(self, cuda):
        self.cuda = cuda


def test_runtime_check_accepts_blackwell_compatible_builds():
    torch_module = FakeTorch(FakeCuda(True, (12, 0), ["sm_90", "sm_120"]))

    validate_cuda_runtime(torch_module, bitsandbytes_version="0.49.0")


def test_runtime_check_rejects_pytorch_without_blackwell_architecture():
    torch_module = FakeTorch(FakeCuda(True, (12, 0), ["sm_90"]))

    with pytest.raises(RuntimeError, match="PyTorch build does not support"):
        validate_cuda_runtime(torch_module, bitsandbytes_version="0.49.0")


def test_runtime_check_rejects_old_bitsandbytes_on_blackwell():
    torch_module = FakeTorch(FakeCuda(True, (12, 0), ["sm_120"]))

    with pytest.raises(RuntimeError, match="bitsandbytes build is too old"):
        validate_cuda_runtime(torch_module, bitsandbytes_version="0.43.1")


def test_runtime_check_allows_cpu_only_imports():
    torch_module = FakeTorch(FakeCuda(False))

    validate_cuda_runtime(torch_module, bitsandbytes_version="0.43.1")
