# GCP Windows Startup Script for SSH Configuration
# This script runs at VM boot to enable SSH access on Windows Server 2022

$ErrorActionPreference = "Stop"

Write-Host "=== Starting GCP SSH Configuration ==="

# Function to fetch metadata from GCP metadata server
function Get-GCPMetadata {
    param([string]$Path)
    try {
        $url = "http://metadata.google.internal/computeMetadata/v1/$Path"
        $headers = @{ "Metadata-Flavor" = "Google" }
        return (Invoke-RestMethod -Uri $url -Headers $headers -TimeoutSec 10).Trim()
    }
    catch {
        Write-Host "Failed to fetch metadata from $Path : $_"
        return $null
    }
}

# Step 1: Install OpenSSH Server if not already installed
Write-Host "Checking OpenSSH Server installation..."
$sshServerFeature = Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'

if ($sshServerFeature.State -ne 'Installed') {
    Write-Host "Installing OpenSSH Server..."
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
    Write-Host "OpenSSH Server installed successfully."
} else {
    Write-Host "OpenSSH Server already installed."
}

# Step 2: Start and enable SSH service
Write-Host "Configuring SSH service..."
Set-Service -Name sshd -StartupType 'Automatic'
Start-Service sshd

# Step 3: Configure Windows Firewall for SSH
Write-Host "Configuring Windows Firewall for SSH..."
$firewallRule = Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue
if (-not $firewallRule) {
    New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' `
        -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
    Write-Host "Firewall rule created for SSH."
} else {
    Write-Host "Firewall rule for SSH already exists."
}

# Step 4: Configure SSH public key authentication
Write-Host "Configuring SSH public key authentication..."

# Fetch SSH public key from GCP metadata
$sshPubKey = Get-GCPMetadata -Path "instance/attributes/ssh_pubkey"

if ($sshPubKey) {
    Write-Host "SSH public key retrieved from metadata."
    
    # Create SSH directory for administrators
    $sshDir = "C:\ProgramData\ssh"
    if (-not (Test-Path $sshDir)) {
        New-Item -ItemType Directory -Path $sshDir -Force | Out-Null
    }
    
    # Write public key to administrators_authorized_keys
    $authKeysFile = Join-Path $sshDir "administrators_authorized_keys"
    Set-Content -Path $authKeysFile -Value $sshPubKey -Force
    Write-Host "SSH public key written to $authKeysFile"
    
    # Set correct ACL for administrators_authorized_keys
    # Only SYSTEM and Administrators should have access
    $acl = Get-Acl $authKeysFile
    $acl.SetAccessRuleProtection($true, $false)
    
    # Remove all existing rules
    $acl.Access | ForEach-Object { $acl.RemoveAccessRule($_) | Out-Null }
    
    # Add SYSTEM with full control
    $systemSid = New-Object System.Security.Principal.SecurityIdentifier("S-1-5-18")
    $systemRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
        $systemSid, "FullControl", "Allow")
    $acl.AddAccessRule($systemRule)
    
    # Add Administrators with full control
    $adminsSid = New-Object System.Security.Principal.SecurityIdentifier("S-1-5-32-544")
    $adminsRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
        $adminsSid, "FullControl", "Allow")
    $acl.AddAccessRule($adminsRule)
    
    Set-Acl -Path $authKeysFile -AclObject $acl
    Write-Host "ACL set correctly for administrators_authorized_keys"
} else {
    Write-Host "WARNING: No SSH public key found in metadata. Password authentication will be required."
}

# Step 5: Configure SSH daemon to allow public key authentication
Write-Host "Configuring sshd_config..."
$sshdConfigPath = "C:\ProgramData\ssh\sshd_config"

if (Test-Path $sshdConfigPath) {
    $sshdConfig = Get-Content $sshdConfigPath
    
    # Ensure PubkeyAuthentication is enabled (not commented out)
    $pubkeyEnabled = $sshdConfig | Where-Object { $_ -match '^\s*PubkeyAuthentication\s+yes' -and $_ -notmatch '^\s*#' }
    if (-not $pubkeyEnabled) {
        Add-Content -Path $sshdConfigPath -Value "`nPubkeyAuthentication yes"
        Write-Host "Enabled PubkeyAuthentication in sshd_config"
    } else {
        Write-Host "PubkeyAuthentication already enabled in sshd_config"
    }
    
    # Ensure PasswordAuthentication is disabled for better security (optional)
    # Uncomment if you want to enforce key-only authentication
    # $passwordDisabled = $sshdConfig | Where-Object { $_ -match '^\s*PasswordAuthentication\s+no' -and $_ -notmatch '^\s*#' }
    # if (-not $passwordDisabled) {
    #     Add-Content -Path $sshdConfigPath -Value "`nPasswordAuthentication no"
    #     Write-Host "Disabled PasswordAuthentication in sshd_config"
    # }
}

# Step 6: Restart SSH service to apply configuration
Write-Host "Restarting SSH service..."
Restart-Service sshd

# Step 7: Verify SSH service is running
$sshdStatus = Get-Service sshd
if ($sshdStatus.Status -eq 'Running') {
    Write-Host "SSH service is running successfully."
} else {
    Write-Host "ERROR: SSH service is not running!"
    exit 1
}

Write-Host "=== GCP SSH Configuration Complete ==="
