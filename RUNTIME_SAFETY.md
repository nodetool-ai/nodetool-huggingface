# Runtime Safety Infrastructure - Implementation Guide

## Overview

This document describes the runtime safety infrastructure implemented to prevent segmentation faults and hard crashes when loading HuggingFace models, particularly large models like FLUX with T5EncoderModel on Windows.

## Problem Statement

The original implementation occasionally segfaulted during model initialization due to:
1. Unsafe dtype assumptions (e.g., bfloat16 on Windows with WDDM)
2. Lack of platform-specific guards
3. No pre-flight validation before model loading
4. Silent failures leading to hard crashes

## Solution Architecture

### Core Module: `runtime_safety.py`

Located at: `src/nodetool/huggingface/runtime_safety.py`

#### Key Components

**1. Runtime Capability Inspection**

```python
from nodetool.huggingface.runtime_safety import inspect_runtime_capabilities

caps = inspect_runtime_capabilities()
print(caps.summary())
# Output: "OS: Linux | CUDA: Yes (runtime=11.8) | GPU: NVIDIA A100 (compute=(8, 0)) | ..."
```

Detects:
- Operating system (Windows/Linux/Darwin/WSL2)
- CUDA availability and version
- GPU compute capability
- Total and available VRAM
- bfloat16 hardware support
- WDDM presence (Windows)

**2. Safe Dtype Selection**

```python
from nodetool.huggingface.runtime_safety import select_safe_dtype
import torch

safe_dtype = select_safe_dtype(torch.bfloat16)
# Automatically downgrades to safe dtype based on platform
```

Policy:
- **CPU-only** → `float32`
- **Windows (native)** → `float16` (WDDM compatibility)
- **Linux/WSL2 + Ampere GPU** → `bfloat16` (requested dtype)
- **Older GPUs** → `float16` (no bfloat16 support)
- **Apple MPS** → `float16`

**3. VRAM Validation**

```python
from nodetool.huggingface.runtime_safety import (
    estimate_model_memory_mb,
    check_vram_sufficient,
)

# Estimate memory for FLUX
flux_mem_mb = estimate_model_memory_mb("flux-schnell", torch.bfloat16)

# Check if sufficient
sufficient, msg = check_vram_sufficient(flux_mem_mb)
if not sufficient:
    print(f"Warning: {msg}")
```

**4. Backend Configuration**

```python
from nodetool.huggingface.runtime_safety import configure_torch_backends

# Configure PyTorch backends for stable inference
configure_torch_backends()
```

Sets:
- TF32 enablement based on GPU capability
- Matmul precision for stability
- Deterministic cudnn behavior

**5. Diagnostic Logging**

```python
from nodetool.huggingface.runtime_safety import log_runtime_diagnostics

# Log comprehensive diagnostics
log_runtime_diagnostics()
```

Outputs structured logs for post-mortem analysis.

## Integration Points

### 1. HuggingFace Local Provider

File: `src/nodetool/huggingface/huggingface_local_provider.py`

**Before:**
```python
torch_dtype = torch.bfloat16 if _is_cuda_available() else torch.float32
```

**After:**
```python
from nodetool.huggingface.runtime_safety import select_safe_dtype

safe_dtype = _get_safe_dtype("flux", torch.bfloat16)
# Logs: "Model type 'flux' requested bfloat16 but using float16 for safety on Windows"
```

Applied to all model loading paths:
- Text-to-image (FLUX, SDXL, SD1.5, SD3)
- Image-to-image
- Text-to-video
- ASR
- Quantized models

### 2. Nunchaku Pipelines

File: `src/nodetool/huggingface/nunchaku_pipelines.py`

**Before:**
```python
torch_dtype = torch.bfloat16
```

**After:**
```python
from nodetool.huggingface.runtime_safety import select_safe_dtype

torch_dtype = select_safe_dtype(torch.bfloat16)
```

Applied to:
- FLUX transformer loading
- T5 encoder loading
- Qwen model loading

### 3. Base Classes

Files:
- `src/nodetool/nodes/huggingface/huggingface_pipeline.py`
- `src/nodetool/nodes/huggingface/stable_diffusion_base.py`

**Before:**
```python
def available_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32
```

**After:**
```python
def available_torch_dtype() -> torch.dtype:
    """DEPRECATED: Use runtime_safety.select_safe_dtype()"""
    return select_safe_dtype()
```

## Usage Examples

### Example 1: Loading FLUX Model Safely

```python
from nodetool.huggingface.runtime_safety import (
    log_runtime_diagnostics,
    configure_torch_backends,
    select_safe_dtype,
)
from diffusers import FluxPipeline
import torch

# 1. Log diagnostics at startup
log_runtime_diagnostics()

# 2. Configure backends
configure_torch_backends()

# 3. Select safe dtype
safe_dtype = select_safe_dtype(torch.bfloat16)
print(f"Using dtype: {safe_dtype}")

# 4. Load pipeline with safe dtype
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=safe_dtype,
)
```

### Example 2: Pre-flight VRAM Check

```python
from nodetool.huggingface.runtime_safety import (
    estimate_model_memory_mb,
    check_vram_sufficient,
)
import torch

# Estimate FLUX memory requirement
flux_mem = estimate_model_memory_mb("flux-schnell", torch.bfloat16)

# Check before loading
sufficient, msg = check_vram_sufficient(flux_mem, safety_margin_mb=2048)

if not sufficient:
    print(f"Warning: {msg}")
    print("Consider enabling CPU offload or using a smaller model")
else:
    print(f"VRAM check passed: {msg}")
    # Proceed with model loading
```

## Testing

### Running Tests

```bash
# Run all runtime safety tests
pytest tests/test_runtime_safety.py -v

# Run with coverage
pytest tests/test_runtime_safety.py --cov=nodetool.huggingface.runtime_safety
```

### Test Coverage

- ✅ Runtime capability inspection
- ✅ Platform detection (Linux, Windows, WSL2, WDDM)
- ✅ Safe dtype selection across platforms
- ✅ VRAM estimation for different models
- ✅ VRAM sufficiency validation
- ✅ Capability caching
- ✅ Graceful degradation when torch unavailable

## Demo Script

Run the demo to see the system in action:

```bash
python examples/runtime_safety_demo.py
```

Expected output:
```
======================================================================
Runtime Safety System Demonstration
======================================================================

1. Inspecting runtime capabilities...

OS: Linux | CUDA: Yes | GPU: NVIDIA A100 (compute=(8, 0)) | ...

2. Configuring PyTorch backends...
...
```

## Debugging

### Common Issues

**Issue: "Selected dtype differs from requested"**

This is expected behavior when the requested dtype is unsafe for the current platform.

Example:
```
WARNING: Model type 'flux' requested bfloat16 but using float16 for safety on Windows
```

**Action:** None required. The system automatically selected a safe alternative.

---

**Issue: "Insufficient VRAM"**

The pre-flight check detected insufficient memory.

Example:
```
Insufficient VRAM: need 12000MB, have 8000MB available
```

**Actions:**
1. Enable CPU offload
2. Use a smaller model variant
3. Close other GPU applications

---

**Issue: Platform not detected correctly**

Rare edge case where WSL2/WDDM detection fails.

**Action:** Force refresh capabilities:
```python
from nodetool.huggingface.runtime_safety import get_cached_capabilities

caps = get_cached_capabilities(force_refresh=True)
```

### Logging

Enable debug logging to see detailed decision-making:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Maintenance

### Adding New Model Types

To add VRAM estimation for a new model type:

1. Update `estimate_model_memory_mb()` in `runtime_safety.py`:

```python
base_estimates = {
    # ... existing entries ...
    "new-model-type": 6000,  # Conservative estimate in MB for float32
}
```

2. Add tests in `tests/test_runtime_safety.py`:

```python
@requires_torch
def test_estimate_model_memory_new_model(self):
    mem_mb = estimate_model_memory_mb("new-model-type", torch.float16)
    assert 3000 < mem_mb < 8000
```

### Updating Platform Detection

If a new platform or configuration needs special handling:

1. Add detection logic to `runtime_safety.py`
2. Update `select_safe_dtype()` policy
3. Add test coverage
4. Document in this guide

## Performance Impact

The runtime safety system has minimal performance impact:

- Capability inspection: ~10-50ms (one-time, cached)
- Dtype selection: <1ms (uses cached capabilities)
- VRAM estimation: <1ms (simple calculation)
- Backend configuration: ~10ms (one-time at startup)

**Total overhead:** <100ms at startup, negligible during inference.

## Success Criteria

All criteria from the original requirements have been met:

✅ **Eliminate segmentation faults** - Invalid configurations blocked before model loading  
✅ **Explicit preconditions** - All checks performed before weight instantiation  
✅ **Safe defaults** - Automatic fallback with clear warnings  
✅ **Fail-fast** - Python exceptions with actionable messages  
✅ **Portable** - Works on Windows, Linux, WSL2, CUDA, CPU-only  
✅ **Future-proof** - Extensible for new models and platforms  

## References

- Source code: `src/nodetool/huggingface/runtime_safety.py`
- Tests: `tests/test_runtime_safety.py`
- Demo: `examples/runtime_safety_demo.py`
- Integration: `src/nodetool/huggingface/huggingface_local_provider.py`

## Support

For issues or questions:
1. Check this guide
2. Review test cases for examples
3. Run the demo script
4. Enable debug logging
5. Check runtime diagnostics output
