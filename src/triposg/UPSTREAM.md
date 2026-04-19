# Vendored: TripoSG

This directory contains a vendored copy of **TripoSG** from
[VAST-AI-Research/TripoSG](https://github.com/VAST-AI-Research/TripoSG).

## Upstream details

| Field | Value |
|---|---|
| Repository | https://github.com/VAST-AI-Research/TripoSG |
| License | MIT |
| HuggingFace model | https://huggingface.co/VAST-AI/TripoSG |

## Why vendored?

TripoSG is not published to PyPI. Rather than requiring users to
`pip install git+https://...`, we vendor the inference-only subset of
the repository so that it ships inside the `nodetool-huggingface` wheel.

Only the files needed at inference time are included — training scripts,
notebooks, and data-processing utilities are omitted.

## Local patches

- None currently applied on top of upstream source.

## Updating

To update this vendored copy:

1. Clone the upstream repo at the desired commit.
2. Copy the inference-relevant directories (`triposg/`) here.
3. Update the table above with the new commit SHA.
4. Run `pytest tests/test_local_3d_smoke.py` to verify nothing broke.
   (Note: 3D nodes are now split across `text_to_3d.py` and `image_to_3d.py`
   with shared helpers in `_3d_common.py`.)
