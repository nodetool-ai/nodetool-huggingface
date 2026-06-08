# Layer the HuggingFace node stack on top of the core worker image.
#
# Rationale: the HF image adds only Python packages on top of nodetool-core.
# `python -m nodetool.worker` comes from the inherited nodetool-core dependency,
# so EXPOSE 7777, HEALTHCHECK, and CMD are all inherited from the core image —
# HF has no worker module of its own.
#
# CUDA strategy: torch 2.9 and the rest of the ML stack pull CUDA-enabled wheels
# from PyPI that bundle their own CUDA runtime. There is therefore no need for an
# nvidia/cuda base image — the host only needs the NVIDIA driver plus a container
# runtime (e.g. the NVIDIA Container Toolkit) at run time. On CPU-only hosts
# (e.g. macOS Docker) the image still runs; GPU-only nodes are expected to fail.
#
# CORE_IMAGE: the locally-built `nodetool-core:local`, or a published
#   ghcr.io/nodetool-ai/nodetool:<tag>. HF_VERSION pins the PyPI release.
ARG CORE_IMAGE=nodetool-core:local
FROM ${CORE_IMAGE}

ARG HF_VERSION=0.7.1
USER root

# Install the released HuggingFace node package from PyPI on top of core.
# Optional extras (ocr, hunyuan3d, triposg, …) are intentionally not installed;
# they become additional build-args / image variants if needed later.
RUN uv pip install \
        --python $VIRTUAL_ENV \
        --index-url https://pypi.org/simple \
        "nodetool-huggingface==${HF_VERSION}" \
    && rm -rf /root/.cache/uv /root/.cache/pip /tmp/* /var/tmp/*

# EXPOSE 7777, HEALTHCHECK, and CMD are all inherited from the core image.
