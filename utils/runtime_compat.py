from importlib.metadata import PackageNotFoundError, version as package_version

from packaging.version import Version


MIN_BLACKWELL_BITSANDBYTES = Version("0.49.0")


def validate_cuda_runtime(torch_module, bitsandbytes_version=None):
    """Reject CUDA builds that cannot execute on an active Blackwell GPU."""
    if not torch_module.cuda.is_available():
        return

    capability = torch_module.cuda.get_device_capability()
    if capability < (12, 0):
        return

    architecture = f"{capability[0]}{capability[1]}"
    supported_architectures = set(torch_module.cuda.get_arch_list())
    if not {
        f"sm_{architecture}",
        f"compute_{architecture}",
    }.intersection(supported_architectures):
        raise RuntimeError(
            "The installed PyTorch build does not support this RTX 50-series "
            f"GPU (sm_{architecture}). Install the project requirements to "
            "restore torch==2.12.0 with CUDA 13.0."
        )

    if bitsandbytes_version is None:
        try:
            bitsandbytes_version = package_version("bitsandbytes")
        except PackageNotFoundError as error:
            raise RuntimeError(
                "bitsandbytes is required for the default 4-bit run. Install "
                "the project requirements to add bitsandbytes==0.49.0."
            ) from error

    if Version(bitsandbytes_version) < MIN_BLACKWELL_BITSANDBYTES:
        raise RuntimeError(
            "The installed bitsandbytes build is too old for this RTX "
            "50-series GPU. Install the project requirements to restore "
            "bitsandbytes==0.49.0."
        )
