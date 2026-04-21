"""Environment checks for notebooks that need specific packages or hardware."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import re
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class PackageRequirement:
    """Import/package requirement with optional version constraint."""

    import_name: str
    distribution_name: str | None = None
    min_version: str | None = None
    display_name: str | None = None

    @property
    def package_name(self) -> str:
        return self.distribution_name or self.import_name

    @property
    def label(self) -> str:
        return self.display_name or self.import_name


def _coerce_package_requirement(
    requirement: str | Mapping[str, Any] | PackageRequirement,
) -> PackageRequirement:
    if isinstance(requirement, PackageRequirement):
        return requirement
    if isinstance(requirement, str):
        return PackageRequirement(import_name=requirement)
    return PackageRequirement(**requirement)


def _distribution_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _numeric_version_parts(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def _version_satisfies(installed: str | None, minimum: str | None) -> bool:
    if minimum is None:
        return True
    if installed is None:
        return False

    try:
        from packaging.version import Version

        return Version(installed) >= Version(minimum)
    except Exception:
        installed_parts = _numeric_version_parts(installed)
        minimum_parts = _numeric_version_parts(minimum)
        return installed_parts >= minimum_parts


def check_package_requirements(
    requirements: Sequence[str | Mapping[str, Any] | PackageRequirement],
) -> list[dict[str, Any]]:
    """Check whether importable packages are installed and satisfy minimum versions."""
    checks = []
    for item in requirements:
        requirement = _coerce_package_requirement(item)
        installed = importlib.util.find_spec(requirement.import_name) is not None
        version = _distribution_version(requirement.package_name) if installed else None
        version_ok = _version_satisfies(version, requirement.min_version)
        checks.append(
            {
                "name": requirement.label,
                "import_name": requirement.import_name,
                "package_name": requirement.package_name,
                "installed": installed,
                "version": version,
                "min_version": requirement.min_version,
                "ok": installed and version_ok,
            }
        )
    return checks


def get_python_environment_info() -> dict[str, str]:
    """Return basic Python and platform information."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def get_cuda_environment_info(device_index: int = 0) -> dict[str, Any]:
    """Return CUDA and GPU information without failing when CUDA is unavailable."""
    cuda_available = torch.cuda.is_available()
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count() if cuda_available else 0,
    }
    if not cuda_available:
        return info

    properties = torch.cuda.get_device_properties(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    info.update(
        {
            "device_index": device_index,
            "gpu_name": properties.name,
            "gpu_total_memory_gb": properties.total_memory / 1024**3,
            "compute_capability": capability,
            "compute_capability_text": f"{capability[0]}.{capability[1]}",
        }
    )
    return info


def _format_package_check(check: Mapping[str, Any]) -> str:
    version = check["version"] or "unknown"
    minimum = check["min_version"]
    suffix = f" (requires >= {minimum})" if minimum else ""
    status = "OK" if check["ok"] else "MISSING"
    if check["installed"] and not check["ok"]:
        status = "OLD"
    return f"{check['name']}: {version}{suffix} [{status}]"


def print_environment_report(report: Mapping[str, Any]) -> None:
    """Print a compact notebook-friendly environment report."""
    python_info = report["python"]
    cuda_info = report["cuda"]

    print(f"Python: {python_info['python']}")
    print(f"Platform: {python_info['platform']}")
    print(f"PyTorch: {cuda_info['torch']}")
    print(f"CUDA available: {cuda_info['cuda_available']}")
    print(f"CUDA: {cuda_info['torch_cuda']}")
    print(f"cuDNN: {cuda_info['cudnn']}")
    if cuda_info["cuda_available"]:
        print(f"GPU: {cuda_info['gpu_name']}")
        print(f"GPU memory: {cuda_info['gpu_total_memory_gb']:.1f} GB")
        print(f"CUDA capability: {cuda_info['compute_capability_text']}")

    packages = report.get("packages", [])
    if packages:
        print("Packages:")
        for check in packages:
            print(f"  - {_format_package_check(check)}")


def environment_report(
    packages: Sequence[str | Mapping[str, Any] | PackageRequirement] = (),
    device_index: int = 0,
) -> dict[str, Any]:
    """Collect Python, package, and CUDA information in one dictionary."""
    return {
        "python": get_python_environment_info(),
        "packages": check_package_requirements(packages),
        "cuda": get_cuda_environment_info(device_index=device_index),
    }


def require_environment(
    packages: Sequence[str | Mapping[str, Any] | PackageRequirement] = (),
    *,
    cuda: bool = False,
    min_cuda_capability: tuple[int, int] | None = None,
    device_index: int = 0,
    error_prefix: str = "Environment check failed",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Validate package and CUDA requirements, print versions, and return the report.

    Args:
        packages: Package names or PackageRequirement objects to check.
        cuda: Require CUDA availability.
        min_cuda_capability: Minimum CUDA compute capability, for example ``(7, 0)``.
        device_index: CUDA device index to inspect.
        error_prefix: Prefix used in raised RuntimeError messages.
        verbose: Print a compact report when checks pass.
    """
    report = environment_report(packages=packages, device_index=device_index)
    errors = []

    failed_packages = [check for check in report["packages"] if not check["ok"]]
    if failed_packages:
        errors.append(
            "missing or incompatible packages: "
            + ", ".join(_format_package_check(check) for check in failed_packages)
        )

    cuda_info = report["cuda"]
    if cuda and not cuda_info["cuda_available"]:
        errors.append("CUDA is required but is not available")

    if min_cuda_capability is not None and cuda_info["cuda_available"]:
        actual = tuple(cuda_info["compute_capability"])
        if actual < min_cuda_capability:
            required = f"{min_cuda_capability[0]}.{min_cuda_capability[1]}"
            errors.append(
                f"CUDA compute capability {required}+ is required; found "
                f"{cuda_info['compute_capability_text']}"
            )

    if errors:
        raise RuntimeError(f"{error_prefix}: " + "; ".join(errors))

    if verbose:
        print_environment_report(report)
    return report
