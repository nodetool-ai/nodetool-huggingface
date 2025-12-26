# GCP Instance Setup for Nodetool HuggingFace

This directory contains scripts to manage a Google Cloud Platform (GCP) instance with CUDA 12.1 for running Stable Diffusion examples. The instance automatically shuts down after 2 hours using GCP's max-run-duration feature.

## Files

- **`setup_gcp_instance.py`**: Python script to manage GCP instances with CUDA support
- **`run_sd_example.py`**: Example script that generates an image using Stable Diffusion XL

## Prerequisites

### For Local Development

1. Install required Python packages:
   ```bash
   pip install google-cloud-compute google-auth
   ```

2. Set up GCP credentials and environment variables:
   ```bash
   export GCP_PROJECT_ID="your-project-id"
   export GCP_ZONE="europe-west4-a"  # Optional, defaults to NL region
   export GCP_ACCOUNT_KEY='{"type": "service_account", ...}'  # Service account JSON
   ```

### For GitHub Actions

1. Configure the following secrets in your GitHub repository:
   - `GCP_PROJECT_ID`: Your Google Cloud project ID
   - `GCP_ACCOUNT_KEY`: Service account key JSON (with Compute Engine permissions)

2. The service account needs the following IAM roles:
   - Compute Instance Admin (v1)
   - Service Account User

## Usage

### Using the Python Script Locally

The script supports three actions: `start`, `run`, and `stop`.

#### Start the instance:
```bash
cd scripts
python setup_gcp_instance.py start
```
This creates a new instance if it doesn't exist, or starts it if it's stopped.

#### Run the SD example script:
```bash
cd scripts
python setup_gcp_instance.py run
```
This executes the Stable Diffusion example on the running instance.

#### Stop the instance:
```bash
cd scripts
python setup_gcp_instance.py stop
```
This stops the running instance.

### Using GitHub Actions

1. Go to the "Actions" tab in your GitHub repository
2. Select "GCP Instance Setup" workflow
3. Click "Run workflow"

The workflow automatically executes: start → run → stop

## What the Setup Does

### Start Action

When you start the instance, the script will:

1. **Check if instance exists** - creates new if needed, starts if stopped
2. **Create a GCP Compute Engine instance** (if needed) with:
   - Deep Learning VM image with CUDA 12.1 pre-installed
   - NVIDIA Tesla T4 GPU (1x)
   - n1-standard-4 machine type (4 vCPUs, 15 GB RAM)
   - 50 GB boot disk
   - **Auto-shutdown after 2 hours** (max-run-duration)
   - NL region (europe-west4-a) by default

3. **Run a startup script** that:
   - Updates system packages
   - Installs Python 3, pip, and git
   - Verifies CUDA and GPU availability
   - Clones the nodetool-huggingface repository
   - Creates a Python virtual environment
   - Installs the project using `pip install -e .`
   - Runs the Stable Diffusion example script

### Run Action

Executes the Stable Diffusion example script on the running instance:
- Model: `nunchaku-tech/nunchaku-sdxl` (quantized SDXL)
- Prompt: "A serene mountain landscape at sunset, highly detailed, photorealistic"
- Output: Saved to `/opt/nodetool-huggingface/nodetool-huggingface/scripts/outputs/sd_example_output.png`

### Stop Action

Stops the running instance to avoid costs.

## Monitoring the Setup

After starting the instance, you can monitor the setup progress:

1. SSH into the instance:
   ```bash
   gcloud compute ssh nodetool-hf-instance --zone europe-west4-a
   ```

2. View the startup script logs:
   ```bash
   tail -f /var/log/startup-script.log
   ```

3. Check the generated image:
   ```bash
   ls -la /opt/nodetool-huggingface/nodetool-huggingface/scripts/outputs/
   ```

## Auto-Shutdown

The instance is configured with a **max-run-duration of 2 hours**. This means:
- The instance will automatically shut down 2 hours after it starts
- This prevents runaway costs if you forget to stop the instance
- The instance can be restarted by running the start command again

## Running the Example Manually

If you want to run the Stable Diffusion example manually on any machine:

```bash
cd scripts
python run_sd_example.py
```

This will:
- Download the quantized SDXL model (if not already cached)
- Generate an image based on the configured prompt
- Save the output to `scripts/outputs/sd_example_output.png`

## Cost Considerations

⚠️ **Important**: Running GCP instances costs money!

- The default configuration (n1-standard-4 + Tesla T4 GPU) costs approximately **$0.50-0.60 per hour**
- **Always stop the instance** when you're done to avoid ongoing charges
- The instance automatically shuts down after 2 hours as a safety measure
- Stop the instance via: `python setup_gcp_instance.py stop`

## Customization

You can customize the instance configuration by modifying `setup_gcp_instance.py`:

- **Machine type**: Change `machine_type` (e.g., `"n1-standard-8"` for more CPU/RAM)
- **GPU type**: Change `gpu_type` (e.g., `"nvidia-tesla-v100"` for more powerful GPU)
- **GPU count**: Change `gpu_count` (e.g., `2` for 2 GPUs)
- **Zone**: Change `zone` parameter or set `GCP_ZONE` environment variable
- **Disk size**: Modify `disk.initialize_params.disk_size_gb`

You can also customize the Stable Diffusion example in `run_sd_example.py`:

- **Model**: Change `sdxl.model.repo_id` and `sdxl.model.path`
- **Prompt**: Modify `sdxl.prompt`
- **Image size**: Adjust `sdxl.height` and `sdxl.width`
- **Quality**: Increase `sdxl.num_inference_steps` (more steps = better quality, slower)

## Troubleshooting

### Instance creation fails

- Check that your service account has the correct permissions
- Verify that the project ID is correct
- Ensure the zone supports GPU instances
- Check GCP quotas for GPU availability

### Startup script fails

- SSH into the instance and check `/var/log/startup-script.log`
- Common issues:
  - CUDA driver mismatch
  - Out of disk space
  - Network connectivity issues
  - Model download failures

### Model download is slow

- The first run downloads several GB of model files
- This can take 10-30 minutes depending on your connection
- Models are cached for subsequent runs

### Out of memory errors

- Try enabling CPU offload: `sdxl.enable_cpu_offload = True` (already enabled by default)
- Use a smaller model or reduce image dimensions
- Upgrade to a larger machine type with more RAM

## Security Notes

- Never commit your `GCP_ACCOUNT_KEY` to the repository
- Use GitHub Secrets for sensitive credentials
- The service account should have minimal required permissions
- Delete instances when not in use to prevent unauthorized access
- Consider using VPC firewall rules to restrict access

## Additional Resources

- [Google Cloud Compute Engine Documentation](https://cloud.google.com/compute/docs)
- [Deep Learning VM Images](https://cloud.google.com/deep-learning-vm/docs/images)
- [Nodetool Documentation](https://docs.nodetool.ai)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
