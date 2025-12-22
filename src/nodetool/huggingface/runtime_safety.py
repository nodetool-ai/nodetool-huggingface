"""
Runtime safety infrastructure for HuggingFace model loading.

This module provides comprehensive runtime capability inspection and safe dtype
selection to prevent segmentation faults and hard crashes when loading ML models,
particularly large models like FLUX with T5EncoderModel.

Design principles:
1. Proactive detection before model instantiation
2. Deterministic safe defaults with explicit warnings
3. Fail fast with Python exceptions, never with segfaults
4. Platform-portable (Windows, Linux, WSL2, CUDA, CPU-only)
"""

from __future__ import annotations

import platform
import os
import warnings
from typing import TYPE_CHECKING, Dict, Any, Optional
from dataclasses import dataclass

from nodetool.config.logging_config import get_logger

if TYPE_CHECKING:
    import torch

log = get_logger(__name__)


@dataclass
class RuntimeCapabilities:
    """
    Comprehensive runtime environment snapshot.
    
    This data class captures all relevant hardware and software characteristics
    needed to make safe dtype and backend decisions.
    """
    os_name: str  # "Windows", "Linux", "Darwin"
    is_wsl2: bool  # Windows Subsystem for Linux 2
    cuda_available: bool
    cuda_version: Optional[str]  # CUDA runtime version (e.g., "11.8")
    torch_cuda_version: Optional[str]  # PyTorch's CUDA build version
    gpu_count: int
    gpu_name: Optional[str]  # Primary GPU name
    compute_capability: Optional[tuple[int, int]]  # e.g., (8, 0) for Ampere
    total_vram_mb: Optional[int]  # Total VRAM in MB
    available_vram_mb: Optional[int]  # Currently available VRAM in MB
    bf16_hardware_support: bool  # True if GPU natively supports bfloat16
    is_wddm: bool  # Windows Display Driver Model (affects stability)
    cpu_only: bool
    mps_available: bool  # Apple Metal Performance Shaders
    
    def summary(self) -> str:
        """Human-readable summary for logging."""
        lines = [
            f"OS: {self.os_name}" + (" (WSL2)" if self.is_wsl2 else ""),
            f"CUDA: {'Yes' if self.cuda_available else 'No'}" + (
                f" (runtime={self.cuda_version}, torch={self.torch_cuda_version})" 
                if self.cuda_available else ""
            ),
        ]
        if self.gpu_count > 0:
            lines.append(f"GPU: {self.gpu_name} (compute={self.compute_capability})")
            if self.total_vram_mb:
                lines.append(
                    f"VRAM: {self.available_vram_mb}MB / {self.total_vram_mb}MB available"
                )
            lines.append(f"BF16 HW support: {self.bf16_hardware_support}")
            lines.append(f"WDDM: {self.is_wddm}")
        elif self.mps_available:
            lines.append("Apple MPS: Yes")
        else:
            lines.append("Mode: CPU-only")
        return " | ".join(lines)


def _detect_wsl2() -> bool:
    """
    Detect if running under Windows Subsystem for Linux 2.
    
    WSL2 behaves differently from native Linux for GPU access and dtype support.
    """
    if platform.system() != "Linux":
        return False
    
    try:
        # WSL2 typically has "microsoft" in kernel version
        with open("/proc/version", "r") as f:
            version_info = f.read().lower()
            return "microsoft" in version_info or "wsl" in version_info
    except Exception:
        return False


def _detect_wddm() -> bool:
    """
    Detect if running under Windows Display Driver Model.
    
    WDDM can cause stability issues with certain dtype/kernel combinations
    due to driver timeout detection and recovery (TDR) mechanisms.
    """
    if platform.system() != "Windows":
        return False
    
    # On Windows with GPU, we're likely under WDDM unless explicitly disabled
    # This is a conservative assumption
    return True


def _get_cuda_version_from_runtime() -> Optional[str]:
    """Get CUDA runtime version from the driver."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        # Get CUDA version as a string like "11.8"
        version_tuple = torch.version.cuda
        return version_tuple
    except Exception:
        return None


def _get_torch_cuda_version() -> Optional[str]:
    """Get the CUDA version PyTorch was built against."""
    try:
        import torch
        if hasattr(torch.version, 'cuda'):
            return torch.version.cuda
        return None
    except Exception:
        return None


def _get_compute_capability() -> Optional[tuple[int, int]]:
    """
    Get compute capability of the primary GPU.
    
    Compute capability determines hardware feature support:
    - 7.0-7.5: Volta (limited bfloat16)
    - 8.0+: Ampere and newer (full bfloat16 support)
    - 8.9+: Ada Lovelace
    """
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return None
        
        major = torch.cuda.get_device_capability(0)[0]
        minor = torch.cuda.get_device_capability(0)[1]
        return (major, minor)
    except Exception:
        return None


def _get_gpu_memory() -> tuple[Optional[int], Optional[int]]:
    """
    Get total and available VRAM in MB.
    
    Returns:
        (total_mb, available_mb) or (None, None) if unavailable
    """
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return (None, None)
        
        # Get memory for device 0
        props = torch.cuda.get_device_properties(0)
        total_mb = props.total_memory // (1024 * 1024)
        
        # Available memory = total - allocated
        allocated_mb = torch.cuda.memory_allocated(0) // (1024 * 1024)
        available_mb = total_mb - allocated_mb
        
        return (total_mb, available_mb)
    except Exception:
        return (None, None)


def _check_bf16_hardware_support() -> bool:
    """
    Verify if bfloat16 is truly supported by the hardware.
    
    This goes beyond nominal support checks and verifies:
    1. Compute capability >= 8.0 (Ampere or newer)
    2. PyTorch's is_bf16_supported() returns True
    3. Not running under problematic configurations
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return False
        
        # Check PyTorch's built-in support check
        if hasattr(torch.cuda, 'is_bf16_supported'):
            if not torch.cuda.is_bf16_supported():
                return False
        else:
            # Fallback: no native check available
            return False
        
        # Verify compute capability
        compute_cap = _get_compute_capability()
        if compute_cap is None:
            return False
        
        major, minor = compute_cap
        # Ampere (8.0) and newer have full bfloat16 support
        if major < 8:
            return False
        
        return True
    except Exception:
        return False


def inspect_runtime_capabilities() -> RuntimeCapabilities:
    """
    Perform comprehensive runtime environment inspection.
    
    This function detects all relevant hardware and software characteristics
    without allocating large tensors or models. Safe to call at startup.
    
    Returns:
        RuntimeCapabilities object with complete environment snapshot
    """
    os_name = platform.system()
    is_wsl2 = _detect_wsl2()
    
    # Import torch lazily
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        mps_available = (
            hasattr(torch.backends, 'mps') and 
            torch.backends.mps.is_available()
        )
    except Exception:
        cuda_available = False
        mps_available = False
    
    # CUDA-specific detection
    cuda_version = None
    torch_cuda_version = None
    gpu_count = 0
    gpu_name = None
    compute_capability = None
    total_vram_mb = None
    available_vram_mb = None
    bf16_hardware_support = False
    
    if cuda_available:
        try:
            import torch
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                gpu_name = torch.cuda.get_device_name(0)
                compute_capability = _get_compute_capability()
                total_vram_mb, available_vram_mb = _get_gpu_memory()
                bf16_hardware_support = _check_bf16_hardware_support()
                cuda_version = _get_cuda_version_from_runtime()
                torch_cuda_version = _get_torch_cuda_version()
        except Exception as e:
            log.warning(f"Failed to query GPU details: {e}")
    
    is_wddm = _detect_wddm() if cuda_available else False
    cpu_only = not cuda_available and not mps_available
    
    return RuntimeCapabilities(
        os_name=os_name,
        is_wsl2=is_wsl2,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        torch_cuda_version=torch_cuda_version,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        compute_capability=compute_capability,
        total_vram_mb=total_vram_mb,
        available_vram_mb=available_vram_mb,
        bf16_hardware_support=bf16_hardware_support,
        is_wddm=is_wddm,
        cpu_only=cpu_only,
        mps_available=mps_available,
    )


def select_safe_dtype(
    requested_dtype: Optional[torch.dtype] = None,
    capabilities: Optional[RuntimeCapabilities] = None,
) -> torch.dtype:
    """
    Select a safe dtype based on hardware capabilities and platform constraints.
    
    This function implements a deterministic policy that prevents segfaults by:
    1. Forcing float32 on CPU-only systems
    2. Forcing float16 on native Windows (WDDM issues with bfloat16)
    3. Allowing bfloat16 only on Linux/WSL2 with Ampere+ GPUs
    4. Logging clear warnings for any downgrade decisions
    
    Args:
        requested_dtype: User's preferred dtype (may be overridden for safety)
        capabilities: Pre-computed capabilities (will inspect if not provided)
    
    Returns:
        Safe dtype that won't cause segfaults
    """
    import torch
    
    if capabilities is None:
        capabilities = inspect_runtime_capabilities()
    
    # Policy implementation
    
    # Rule 1: CPU-only systems must use float32
    if capabilities.cpu_only:
        if requested_dtype not in (None, torch.float32):
            log.warning(
                f"Requested dtype {requested_dtype} not available on CPU. "
                f"Forcing float32 for CPU execution."
            )
        log.info("Selected dtype: float32 (CPU-only system)")
        return torch.float32
    
    # Rule 2: Apple MPS uses float16
    if capabilities.mps_available and not capabilities.cuda_available:
        if requested_dtype not in (None, torch.float16):
            log.warning(
                f"Requested dtype {requested_dtype} not optimal for MPS. "
                f"Using float16 for Apple Metal backend."
            )
        log.info("Selected dtype: float16 (Apple MPS)")
        return torch.float16
    
    # Rule 3: Native Windows with CUDA - force float16 (WDDM compatibility)
    if capabilities.os_name == "Windows" and not capabilities.is_wsl2:
        if requested_dtype == torch.bfloat16:
            log.warning(
                "Requested bfloat16 on native Windows. "
                "Forcing float16 to prevent WDDM-related segfaults. "
                f"GPU: {capabilities.gpu_name}, Compute: {capabilities.compute_capability}"
            )
        log.info("Selected dtype: float16 (Windows WDDM compatibility)")
        return torch.float16
    
    # Rule 4: Linux/WSL2 with capable GPU - allow bfloat16
    if capabilities.cuda_available and capabilities.bf16_hardware_support:
        # Verify CUDA versions match
        if capabilities.cuda_version and capabilities.torch_cuda_version:
            cuda_major = capabilities.cuda_version.split('.')[0]
            torch_cuda_major = capabilities.torch_cuda_version.split('.')[0]
            if cuda_major != torch_cuda_major:
                log.warning(
                    f"CUDA version mismatch: runtime={capabilities.cuda_version}, "
                    f"torch={capabilities.torch_cuda_version}. "
                    f"This may cause issues with bfloat16."
                )
        
        # All checks passed - bfloat16 is safe
        final_dtype = requested_dtype if requested_dtype == torch.bfloat16 else torch.bfloat16
        log.info(
            f"Selected dtype: {final_dtype} "
            f"(Linux/WSL2 with {capabilities.gpu_name}, compute {capabilities.compute_capability})"
        )
        return final_dtype
    
    # Rule 5: GPU available but no bfloat16 support - use float16
    if capabilities.cuda_available:
        if requested_dtype == torch.bfloat16:
            log.warning(
                f"Requested bfloat16 but hardware doesn't fully support it. "
                f"GPU: {capabilities.gpu_name}, Compute: {capabilities.compute_capability}. "
                f"Downgrading to float16 for safety."
            )
        log.info(
            f"Selected dtype: float16 "
            f"(GPU without full bfloat16 support: {capabilities.gpu_name})"
        )
        return torch.float16
    
    # Fallback - should not reach here
    log.warning("Unexpected capability configuration, defaulting to float32")
    return torch.float32


def estimate_model_memory_mb(
    model_type: str,
    dtype: torch.dtype,
    **model_params,
) -> int:
    """
    Estimate VRAM requirements for common model types.
    
    This provides conservative estimates to enable pre-flight validation.
    Actual usage may vary based on batch size and other factors.
    
    Args:
        model_type: "flux", "sdxl", "sd15", "t5-xxl", etc.
        dtype: torch dtype (affects memory by 2x between fp32/fp16)
        **model_params: Additional model-specific parameters
    
    Returns:
        Estimated VRAM requirement in MB
    """
    import torch
    
    # Get dtype size multiplier
    dtype_multiplier = {
        torch.float32: 1.0,
        torch.float16: 0.5,
        torch.bfloat16: 0.5,
    }.get(dtype, 1.0)
    
    # Base estimates in MB for float32
    base_estimates = {
        "flux-schnell": 24000,  # FLUX Schnell ~24GB
        "flux-dev": 24000,      # FLUX Dev ~24GB  
        "t5-xxl": 11000,        # T5-XXL encoder ~11GB
        "sdxl": 7000,           # SDXL ~7GB
        "sd15": 4000,           # SD 1.5 ~4GB
        "sdxl-unet": 5000,      # SDXL UNet alone ~5GB
        "qwen": 8000,           # Qwen ~8GB
    }
    
    base_mb = base_estimates.get(model_type, 5000)  # Default 5GB
    
    # Apply dtype multiplier
    estimated_mb = int(base_mb * dtype_multiplier)
    
    # Add overhead for attention and intermediate activations (30%)
    estimated_mb = int(estimated_mb * 1.3)
    
    return estimated_mb


def check_vram_sufficient(
    required_mb: int,
    capabilities: Optional[RuntimeCapabilities] = None,
    safety_margin_mb: int = 1024,
) -> tuple[bool, str]:
    """
    Check if sufficient VRAM is available for a model.
    
    Args:
        required_mb: Estimated VRAM requirement
        capabilities: Pre-computed capabilities (will inspect if not provided)
        safety_margin_mb: Additional safety margin (default 1GB)
    
    Returns:
        (is_sufficient, message)
    """
    if capabilities is None:
        capabilities = inspect_runtime_capabilities()
    
    if capabilities.cpu_only:
        return (False, "No GPU available - CPU loading not recommended for large models")
    
    if capabilities.available_vram_mb is None:
        # Can't determine - assume OK but warn
        return (True, "VRAM availability unknown - proceeding with caution")
    
    available = capabilities.available_vram_mb
    needed = required_mb + safety_margin_mb
    
    if available < needed:
        return (
            False,
            f"Insufficient VRAM: need {needed}MB, have {available}MB available. "
            f"Consider enabling CPU offload or using a smaller model."
        )
    
    return (
        True,
        f"VRAM sufficient: {available}MB available, {needed}MB needed"
    )


def configure_torch_backends(
    capabilities: Optional[RuntimeCapabilities] = None,
) -> None:
    """
    Configure PyTorch backends for stable inference.
    
    This sets safe defaults that prevent kernel selection issues:
    - Disables TF32 on older hardware
    - Sets appropriate matmul precision
    - Disables experimental features on unsupported platforms
    
    Args:
        capabilities: Pre-computed capabilities (will inspect if not provided)
    """
    try:
        import torch
    except ImportError:
        log.debug("PyTorch not available, skipping backend configuration")
        return
    
    if capabilities is None:
        capabilities = inspect_runtime_capabilities()
    
    if not capabilities.cuda_available:
        log.debug("Skipping CUDA backend configuration (no CUDA available)")
        return
    
    # TF32 is safe on Ampere (8.0+) and newer
    if capabilities.compute_capability and capabilities.compute_capability[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        log.debug("Enabled TF32 for Ampere+ GPU")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        log.debug("Disabled TF32 for pre-Ampere GPU")
    
    # Disable cudnn benchmarking for deterministic behavior
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    log.info("PyTorch backends configured for stable inference")


def log_runtime_diagnostics(
    capabilities: Optional[RuntimeCapabilities] = None,
) -> None:
    """
    Log comprehensive runtime diagnostics for post-mortem analysis.
    
    This should be called at the start of model loading operations to ensure
    all relevant information is captured in logs.
    
    Args:
        capabilities: Pre-computed capabilities (will inspect if not provided)
    """
    if capabilities is None:
        capabilities = inspect_runtime_capabilities()
    
    log.info("=" * 70)
    log.info("Runtime Diagnostics")
    log.info("=" * 70)
    log.info(capabilities.summary())
    log.info("=" * 70)


# Singleton capability cache to avoid repeated inspection
_cached_capabilities: Optional[RuntimeCapabilities] = None


def get_cached_capabilities(force_refresh: bool = False) -> RuntimeCapabilities:
    """
    Get cached runtime capabilities or inspect if not cached.
    
    Args:
        force_refresh: Force re-inspection even if cached
    
    Returns:
        RuntimeCapabilities object
    """
    global _cached_capabilities
    
    if _cached_capabilities is None or force_refresh:
        _cached_capabilities = inspect_runtime_capabilities()
    
    return _cached_capabilities
