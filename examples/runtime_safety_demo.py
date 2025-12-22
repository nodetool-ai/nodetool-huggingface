#!/usr/bin/env python3
"""
Example demonstrating the runtime safety system.

This script shows how the runtime capability inspection and safe dtype
selection work in practice.
"""

from nodetool.huggingface.runtime_safety import (
    inspect_runtime_capabilities,
    select_safe_dtype,
    log_runtime_diagnostics,
    configure_torch_backends,
    estimate_model_memory_mb,
    check_vram_sufficient,
)


def main():
    print("=" * 70)
    print("Runtime Safety System Demonstration")
    print("=" * 70)
    print()
    
    # 1. Inspect runtime capabilities
    print("1. Inspecting runtime capabilities...")
    print()
    log_runtime_diagnostics()
    print()
    
    # 2. Configure torch backends for stable inference
    print("2. Configuring PyTorch backends...")
    configure_torch_backends()
    print()
    
    # 3. Demonstrate safe dtype selection
    print("3. Safe dtype selection examples:")
    print()
    
    try:
        import torch
        
        # Show what would be selected for different requested dtypes
        for requested in [None, torch.float32, torch.float16, torch.bfloat16]:
            safe = select_safe_dtype(requested)
            print(f"   Requested: {requested} -> Safe: {safe}")
        print()
        
        # 4. Estimate model memory requirements
        print("4. Model memory estimation:")
        print()
        
        models = [
            ("flux-schnell", torch.bfloat16),
            ("sdxl", torch.float16),
            ("sd15", torch.float16),
        ]
        
        for model_type, dtype in models:
            mem_mb = estimate_model_memory_mb(model_type, dtype)
            mem_gb = mem_mb / 1024
            print(f"   {model_type} ({dtype}): ~{mem_gb:.1f}GB")
        print()
        
        # 5. Check VRAM sufficiency
        print("5. VRAM sufficiency check:")
        print()
        
        flux_mem = estimate_model_memory_mb("flux-schnell", torch.bfloat16)
        sufficient, msg = check_vram_sufficient(flux_mem)
        print(f"   FLUX.1-schnell: {msg}")
        print()
        
    except ImportError:
        print("   (torch not available, skipping dtype demonstrations)")
        print()
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("The runtime safety system ensures:")
    print("  ✓ No segfaults from invalid dtype/platform combinations")
    print("  ✓ Deterministic behavior across runs")
    print("  ✓ Clear logging of all decisions")
    print("  ✓ Fail-fast with Python exceptions, not hard crashes")
    print("  ✓ Portable across Windows, Linux, WSL2, and macOS")
    print()


if __name__ == "__main__":
    main()
