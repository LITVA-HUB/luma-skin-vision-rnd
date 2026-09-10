import importlib.metadata
import platform
import subprocess
import sys


def inspect_environment():
    result = {
        "os": platform.platform(),
        "python": sys.version,
        "cpu": platform.processor(),
        "packages": {},
    }
    for name in (
        "numpy",
        "pillow",
        "torch",
        "torchvision",
        "pydantic",
        "opencv-python-headless",
        "onnx",
        "onnxruntime",
    ):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    try:
        import torch

        result.update(cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda)
        if torch.cuda.is_available():
            result.update(
                gpu=torch.cuda.get_device_name(0),
                vram_bytes=torch.cuda.get_device_properties(0).total_memory,
            )
    except ImportError:
        result["cuda_available"] = False
    try:
        result["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        result["nvidia_smi"] = None
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total_phys",
                    "avail_phys",
                    "total_page",
                    "avail_page",
                    "total_virtual",
                    "avail_virtual",
                    "avail_ext",
                )
            ]

        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
        result["ram_bytes"] = memory.total_phys
    return result
