# Windows CI Scripts for GCP GPU Testing

This directory contains PowerShell scripts for running CI tests on Windows Server 2022 with NVIDIA GPUs in Google Cloud Platform.

## Files

### gcp_startup_ssh.ps1
**Purpose**: Windows startup script that configures OpenSSH Server for remote access.

**What it does:**
- Installs OpenSSH Server Windows capability if not present
- Starts and enables the SSH service (sshd)
- Configures Windows Firewall to allow port 22
- Retrieves SSH public key from GCP metadata
- Sets up public key authentication in `C:\ProgramData\ssh\administrators_authorized_keys`
- Configures proper ACLs (SYSTEM and Administrators only)
- Enables PubkeyAuthentication in sshd_config
- Restarts SSH service to apply changes

**Usage**: Automatically injected into VM metadata by the GitHub Actions workflow. Not meant to be run manually.

### gpu_entry.ps1
**Purpose**: Entry point script that validates GPU availability and can be extended to run package tests.

**What it does:**
- Tests NVIDIA GPU availability using nvidia-smi
- Displays GPU configuration (name, driver version, memory, compute capability)
- Checks for Python installation (optional)
- Tests PyTorch CUDA availability if PyTorch is installed (optional)
- Returns exit code 0 on success, 1 on failure

**Usage**: 
```powershell
# Run locally
powershell -NoProfile -ExecutionPolicy Bypass -File gpu_entry.ps1

# Or via GitHub Actions workflow (default entry script)
# The workflow will execute this automatically
```

**Extending for package testing:**
To add actual package tests, modify this script to:
1. Install Python and dependencies
2. Install nodetool-huggingface package
3. Run pytest or other test commands
4. Return appropriate exit codes

## Example: Adding Package Tests

To extend `gpu_entry.ps1` for full package testing:

```powershell
# After GPU checks, add:

Write-Host "`n--- Installing Python and Dependencies ---"
# Install Python
choco install python311 -y
$env:PATH += ";C:\Python311;C:\Python311\Scripts"

# Install package
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .

Write-Host "`n--- Running Package Tests ---"
pytest tests/ -v --tb=short

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Tests failed"
    exit 1
}

Write-Host "SUCCESS: All tests passed"
exit 0
```

## GitHub Actions Workflow

The main workflow file is located at `.github/workflows/gcp-windows-gpu-script.yml`.

**Manual trigger**: Go to Actions → GCP Windows GPU Script Runner → Run workflow

**Key features:**
- Creates ephemeral Windows Server 2022 VM with NVIDIA T4 GPU
- Automatically installs NVIDIA driver if not present
- Handles Windows reboots during driver installation
- Streams logs to GitHub Actions
- Always cleans up resources (VM, firewall rules)
- Fails workflow if script returns non-zero exit code

**Required secrets** (set in repository Settings → Secrets):
- `GCP_PROJECT_ID`: Your GCP project ID
- `GCP_ZONE`: GCP zone (e.g., "us-central1-a")
- `GCP_SA_KEY_JSON`: Service account JSON key
- `GCP_VM_SSH_PUBKEY`: SSH public key
- `GCP_VM_SSH_PRIVKEY`: SSH private key

## Troubleshooting

### SSH connection fails
- Check that GCP_VM_SSH_PUBKEY and GCP_VM_SSH_PRIVKEY match
- Verify firewall rules allow port 22
- Check serial console output in GCP for startup script errors
- Ensure service account has necessary permissions

### NVIDIA driver installation fails
- Check NVIDIA_DRIVER_URL in workflow file is accessible
- Verify GPU type is compatible with the driver version
- Check Windows event logs via serial console
- Ensure adequate disk space (100GB default should be sufficient)

### Script execution fails
- Verify script path is correct relative to repo root
- Check PowerShell execution policy is bypassed in workflow
- Review logs for specific error messages
- Test script locally on Windows before running in CI

## Additional Resources

- [GCP Windows Images](https://cloud.google.com/compute/docs/images/os-details#windows_server)
- [GCP GPU Documentation](https://cloud.google.com/compute/docs/gpus)
- [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- [OpenSSH on Windows](https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse)
