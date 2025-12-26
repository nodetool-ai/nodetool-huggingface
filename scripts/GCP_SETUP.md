# GCP Instance Setup for Nodetool HuggingFace

This directory contains scripts to set up a Google Cloud Platform (GCP) instance with CUDA 12.8, install the nodetool-huggingface project, and run Stable Diffusion examples.

## Files

- **`setup_gcp_instance.py`**: Python script to create/delete GCP instances with CUDA support
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
   export GCP_ZONE="us-central1-a"  # Optional, defaults to us-central1-a
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

#### Create an instance:
```bash
cd scripts
python setup_gcp_instance.py create
```

#### Delete an instance:
```bash
cd scripts
python setup_gcp_instance.py delete
```

### Using GitHub Actions

1. Go to the "Actions" tab in your GitHub repository
2. Select "GCP Instance Setup" workflow
3. Click "Run workflow"
4. Choose the action:
   - **create**: Creates a new instance and runs the setup
   - **delete**: Deletes the instance
5. Optionally specify the GCP zone (defaults to `us-central1-a`)

## What the Setup Does

When you create an instance, the script will:

1. **Create a GCP Compute Engine instance** with:
   - Deep Learning VM image with CUDA 12.8 pre-installed
   - NVIDIA Tesla T4 GPU (1x)
   - n1-standard-4 machine type (4 vCPUs, 15 GB RAM)
   - 50 GB boot disk

2. **Run a startup script** that:
   - Updates system packages
   - Installs Python 3, pip, and git
   - Verifies CUDA and GPU availability
   - Clones the nodetool-huggingface repository
   - Creates a Python virtual environment
   - Installs the project using `pip install -e .`
   - Runs the Stable Diffusion example script

3. **Generate a test image** using:
   - Model: `nunchaku-tech/nunchaku-sdxl` (quantized SDXL)
   - Prompt: "A serene mountain landscape at sunset, highly detailed, photorealistic"
   - Output: Saved to `/opt/nodetool-huggingface/nodetool-huggingface/scripts/outputs/sd_example_output.png`

## Monitoring the Setup

After creating an instance, you can monitor the setup progress:

1. SSH into the instance:
   ```bash
   gcloud compute ssh nodetool-hf-instance --zone us-central1-a
   ```

2. View the startup script logs:
   ```bash
   tail -f /var/log/startup-script.log
   ```

3. Check the generated image:
   ```bash
   ls -la /opt/nodetool-huggingface/nodetool-huggingface/scripts/outputs/
   ```

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
- **Always delete the instance** when you're done to avoid ongoing charges
- You can delete the instance via:
  - The Python script: `python setup_gcp_instance.py delete`
  - The GitHub Actions workflow (choose "delete" action)
  - The GCP Console

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
