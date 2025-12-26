#!/usr/bin/env python3
"""
Script to setup a GCP instance with CUDA 12.8, install nodetool-huggingface,
download a Stable Diffusion model, and run an example.

This script uses the Google Cloud Python SDK to:
1. Create a GCP Compute Engine instance with CUDA 12.8 pre-installed
2. Install the nodetool-huggingface project
3. Download a recommended Stable Diffusion model
4. Run a Stable Diffusion example

Requirements:
- google-cloud-compute
- google-auth

Environment variables:
- GCP_PROJECT_ID: Google Cloud Project ID
- GCP_ZONE: Zone to create the instance (e.g., us-central1-a)
- GCP_ACCOUNT_KEY: Service account key JSON (for authentication)
"""

import json
import os
import sys
import time

try:
    from google.cloud import compute_v1
    from google.oauth2 import service_account
except ImportError:
    print("Error: google-cloud-compute is not installed.")
    print("Install it with: pip install google-cloud-compute google-auth")
    sys.exit(1)


class GCPInstanceSetup:
    """Handles GCP instance creation and setup."""

    def __init__(
        self,
        project_id: str,
        zone: str = "europe-west4-a",
        instance_name: str = "nodetool-hf-instance",
        machine_type: str = "n1-standard-4",
        gpu_type: str = "nvidia-tesla-t4",
        gpu_count: int = 1,
        credentials=None,
    ):
        self.project_id = project_id
        self.zone = zone
        self.instance_name = instance_name
        self.machine_type = machine_type
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
        self.credentials = credentials

        # Initialize clients with credentials
        self.instances_client = compute_v1.InstancesClient(credentials=credentials)
        self.images_client = compute_v1.ImagesClient(credentials=credentials)

    def get_cuda_image(self) -> str:
        """Get the latest Deep Learning VM image with CUDA 12.8."""
        # Deep Learning VM images from Google with CUDA pre-installed
        # These are maintained by Google and include CUDA 12.x
        project = "deeplearning-platform-release"
        family = "common-cu128"

        try:
            # Try to get the image from the family
            image = self.images_client.get_from_family(project=project, family=family)
            print(f"Using image: {image.self_link}")
            return image.self_link
        except Exception as e:
            print(f"Warning: Could not find specific CUDA 12.8 image: {e}")
            # Fallback to a known working Deep Learning image
            return f"projects/{project}/global/images/family/pytorch-latest-gpu"

    def create_startup_script(self) -> str:
        """Create the startup script for the instance."""
        script = """#!/bin/bash
set -e

# Log everything
exec > >(tee /var/log/startup-script.log)
exec 2>&1

echo "Starting instance setup at $(date)"

# Update system
echo "Updating system packages..."
apt-get update
apt-get install -y git python3-pip python3-venv

# Verify CUDA installation
echo "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "Warning: CUDA not found in PATH"
fi

# Verify GPU
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "Warning: nvidia-smi not found"
fi

# Create a working directory
WORK_DIR="/opt/nodetool-huggingface"
mkdir -p "$WORK_DIR"

# Clone the repository if it does not already exist
REPO_DIR="$WORK_DIR/nodetool-huggingface"
echo "Ensuring repository is present at $REPO_DIR..."
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/nodetool-ai/nodetool-huggingface.git "$REPO_DIR"
fi

# Change into the repository directory
cd "$REPO_DIR"

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the project
echo "Installing nodetool-huggingface..."
pip install -e .

# Run the Stable Diffusion example
echo "Running Stable Diffusion example..."
cd scripts
python3 run_sd_example.py

echo "Setup completed successfully at $(date)"
"""
        return script

    def instance_exists(self) -> bool:
        """Check if the instance already exists."""
        try:
            self.instances_client.get(
                project=self.project_id,
                zone=self.zone,
                instance=self.instance_name,
            )
            return True
        except Exception:
            return False

    def get_instance_status(self) -> str:
        """Get the current status of the instance."""
        try:
            instance = self.instances_client.get(
                project=self.project_id,
                zone=self.zone,
                instance=self.instance_name,
            )
            return instance.status
        except Exception:
            return "NOT_FOUND"

    def start_instance(self) -> None:
        """Start a stopped instance."""
        if not self.instance_exists():
            print(f"Error: Instance '{self.instance_name}' does not exist")
            sys.exit(1)
        
        status = self.get_instance_status()
        if status == "RUNNING":
            print(f"Instance '{self.instance_name}' is already running!")
            self._print_instance_info()
            return
        
        print(f"Starting instance '{self.instance_name}'...")
        self._start_instance()

    def stop_instance(self) -> None:
        """Stop a running instance."""
        if not self.instance_exists():
            print(f"Error: Instance '{self.instance_name}' does not exist")
            sys.exit(1)
        
        status = self.get_instance_status()
        if status == "TERMINATED" or status == "STOPPED":
            print(f"Instance '{self.instance_name}' is already stopped!")
            return
        
        print(f"Stopping instance '{self.instance_name}'...")
        try:
            operation = self.instances_client.stop(
                project=self.project_id,
                zone=self.zone,
                instance=self.instance_name,
            )

            print("Waiting for instance to stop...")
            self.wait_for_operation(operation)
            print(f"Instance '{self.instance_name}' stopped successfully!")

        except Exception as e:
            print(f"Error stopping instance: {e}")
            raise

    def run_script(self) -> None:
        """Run the SD example script on an existing running instance."""
        if not self.instance_exists():
            print(f"Error: Instance '{self.instance_name}' does not exist")
            print("Please start the instance first with 'start' command")
            sys.exit(1)
        
        status = self.get_instance_status()
        if status != "RUNNING":
            print(f"Error: Instance '{self.instance_name}' is not running (status: {status})")
            print("Please start the instance first with 'start' command")
            sys.exit(1)
        
        print(f"Running SD example script on instance '{self.instance_name}'...")
        print("\nTo run the script, SSH into the instance and execute:")
        print(f"  gcloud compute ssh {self.instance_name} --zone {self.zone}")
        print("  cd /opt/nodetool-huggingface/nodetool-huggingface/scripts")
        print("  source ../venv/bin/activate")
        print("  python run_sd_example.py")
        print("\nOr run it directly:")
        print(f"  gcloud compute ssh {self.instance_name} --zone {self.zone} --command 'cd /opt/nodetool-huggingface/nodetool-huggingface && source venv/bin/activate && cd scripts && python run_sd_example.py'")

    def _start_instance(self) -> None:
        """Start a stopped instance (internal helper)."""
        try:
            operation = self.instances_client.start(
                project=self.project_id,
                zone=self.zone,
                instance=self.instance_name,
            )

            print("Waiting for instance to start...")
            self.wait_for_operation(operation)
            print(f"Instance '{self.instance_name}' started successfully!")
            self._print_instance_info()

        except Exception as e:
            print(f"Error starting instance: {e}")
            raise

    def _print_instance_info(self) -> None:
        """Print instance information."""
        try:
            instance = self.instances_client.get(
                project=self.project_id,
                zone=self.zone,
                instance=self.instance_name,
            )

            # Get external IP
            if instance.network_interfaces:
                if instance.network_interfaces[0].access_configs:
                    external_ip = (
                        instance.network_interfaces[0].access_configs[0].nat_ip
                    )
                    print(f"External IP: {external_ip}")

            print("\nInstance setup is running in the background.")
            print("You can check the startup script logs with:")
            print(
                f"  gcloud compute ssh {self.instance_name} --zone {self.zone} --command 'tail -f /var/log/startup-script.log'"
            )
        except Exception as e:
            print(f"Warning: Could not retrieve instance info: {e}")

    def create_instance(self) -> None:
        """Create a new GCP instance with CUDA and GPU."""
        print(f"Creating instance '{self.instance_name}' in {self.zone}...")

        # Get the CUDA image
        image_uri = self.get_cuda_image()

        # Configure the instance
        machine_type_full = f"zones/{self.zone}/machineTypes/{self.machine_type}"

        # Configure disks
        disk = compute_v1.AttachedDisk()
        disk.boot = True
        disk.auto_delete = True
        disk.initialize_params = compute_v1.AttachedDiskInitializeParams()
        disk.initialize_params.source_image = image_uri
        disk.initialize_params.disk_size_gb = 50  # 50GB boot disk

        # Configure network
        network_interface = compute_v1.NetworkInterface()
        network_interface.network = (
            f"projects/{self.project_id}/global/networks/default"
        )
        # Add external IP for access
        access_config = compute_v1.AccessConfig()
        access_config.name = "External NAT"
        access_config.type_ = "ONE_TO_ONE_NAT"
        network_interface.access_configs = [access_config]

        # Configure GPU
        accelerator = compute_v1.AcceleratorConfig()
        accelerator.accelerator_count = self.gpu_count
        accelerator.accelerator_type = (
            f"zones/{self.zone}/acceleratorTypes/{self.gpu_type}"
        )

        # Create instance configuration
        instance = compute_v1.Instance()
        instance.name = self.instance_name
        instance.machine_type = machine_type_full
        instance.disks = [disk]
        instance.network_interfaces = [network_interface]
        instance.guest_accelerators = [accelerator]

        # Set scheduling to allow GPU and auto-shutdown
        instance.scheduling = compute_v1.Scheduling()
        instance.scheduling.on_host_maintenance = "TERMINATE"
        instance.scheduling.automatic_restart = True
        # Set max run duration to 2 hours (7200 seconds) for auto-shutdown
        instance.scheduling.max_run_duration = compute_v1.Duration()
        instance.scheduling.max_run_duration.seconds = 7200

        # Add startup script
        metadata = compute_v1.Metadata()
        metadata_item = compute_v1.Items()
        metadata_item.key = "startup-script"
        metadata_item.value = self.create_startup_script()
        metadata.items = [metadata_item]
        instance.metadata = metadata

        # Create the instance
        try:
            operation = self.instances_client.insert(
                project=self.project_id,
                zone=self.zone,
                instance_resource=instance,
            )

            print(f"Creating instance... (Operation: {operation.name})")
            print("Waiting for instance creation to complete...")

            # Wait for operation to complete
            self.wait_for_operation(operation)

            print(f"Instance '{self.instance_name}' created successfully!")
            self._print_instance_info()

        except Exception as e:
            print(f"Error creating instance: {e}")
            raise

    def wait_for_operation(
        self, operation: compute_v1.Operation, timeout: int = 300
    ) -> None:
        """Wait for a GCP operation to complete."""
        start_time = time.time()

        while operation.status != compute_v1.Operation.Status.DONE:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds")

            time.sleep(5)
            # Refresh operation status using credentials stored during initialization
            zone_operations_client = compute_v1.ZoneOperationsClient(
                credentials=self.credentials
            )
            operation = zone_operations_client.get(
                project=self.project_id,
                zone=self.zone,
                operation=operation.name,
            )
            print(".", end="", flush=True)

        print()  # New line after dots

        if operation.error:
            raise Exception(f"Operation failed: {operation.error}")


def main():
    """Main entry point."""
    # Get configuration from environment
    project_id = os.environ.get("GCP_PROJECT_ID")
    zone = os.environ.get("GCP_ZONE", "europe-west4-a")
    service_account_key = os.environ.get("GCP_ACCOUNT_KEY")

    if not project_id:
        print("Error: GCP_PROJECT_ID environment variable not set")
        sys.exit(1)

    if not service_account_key:
        print("Error: GCP_ACCOUNT_KEY environment variable not set")
        sys.exit(1)

    # Set up credentials
    try:
        credentials_dict = json.loads(service_account_key)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict
        )

    except json.JSONDecodeError as e:
        print(f"Error parsing GCP_ACCOUNT_KEY JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error setting up credentials: {e}")
        sys.exit(1)

    # Parse command line arguments
    action = sys.argv[1] if len(sys.argv) > 1 else "start"

    if action not in ["start", "run", "stop"]:
        print(f"Usage: {sys.argv[0]} [start|run|stop]")
        print("\nActions:")
        print("  start - Create (if needed) and start the instance")
        print("  run   - Run the SD example script on the running instance")
        print("  stop  - Stop the running instance")
        sys.exit(1)

    # Create setup manager with credentials
    setup = GCPInstanceSetup(
        project_id=project_id,
        zone=zone,
        credentials=credentials,
    )

    # Execute action
    if action == "start":
        # Check if instance exists, create if not, start if stopped
        if not setup.instance_exists():
            setup.create_instance()
        else:
            setup.start_instance()
    elif action == "run":
        setup.run_script()
    elif action == "stop":
        setup.stop_instance()


if __name__ == "__main__":
    main()
