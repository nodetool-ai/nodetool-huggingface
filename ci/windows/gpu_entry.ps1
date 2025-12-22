# GPU Entry Script for nodetool-huggingface CI Testing
# This script tests GPU availability and can be extended to run package tests

$ErrorActionPreference = "Stop"

Write-Host "=== Starting GPU CI Entry Script ==="

# Test 1: Check NVIDIA GPU availability
Write-Host "`n--- Test 1: NVIDIA GPU Check ---"
try {
    $nvidiaSmiOutput = & nvidia-smi
    Write-Host $nvidiaSmiOutput
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: nvidia-smi failed with exit code $LASTEXITCODE"
        exit 1
    }
    
    Write-Host "SUCCESS: NVIDIA GPU detected and nvidia-smi working"
} catch {
    Write-Host "ERROR: Failed to run nvidia-smi: $_"
    exit 1
}

# Test 2: Check Python installation (if available)
Write-Host "`n--- Test 2: Python Check ---"
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "Python version: $pythonVersion"
} catch {
    Write-Host "WARNING: Python not found or not in PATH. This is expected on fresh VM."
    Write-Host "For actual package testing, Python would be installed here."
}

# Test 3: Check PyTorch CUDA availability (if Python/PyTorch installed)
Write-Host "`n--- Test 3: PyTorch CUDA Check (Optional) ---"
try {
    $cudaCheck = & python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1
    Write-Host $cudaCheck
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: PyTorch CUDA check failed. This is expected if PyTorch is not installed."
    }
} catch {
    Write-Host "WARNING: PyTorch not installed. This is expected on fresh VM."
    Write-Host "For package testing, install Python/PyTorch using CI setup scripts."
}

# Test 4: Display GPU Memory and Configuration
Write-Host "`n--- Test 4: GPU Memory and Configuration ---"
try {
    $gpuQuery = & nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_cap --format=csv
    Write-Host $gpuQuery
} catch {
    Write-Host "WARNING: Could not query detailed GPU information"
}

Write-Host "`n=== GPU CI Entry Script Complete ==="
Write-Host "All critical tests passed successfully!"
exit 0
