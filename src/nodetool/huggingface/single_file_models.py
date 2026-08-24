"""Identify the model family of a single-file checkpoint on disk.

NodeTool's existing single-file support is Hub-shaped: it takes a repo id plus a
filename inside that repo, downloads the file, and decides which diffusers
pipeline to use by reading the repo's *tags* from the Hub API.  None of that is
available for a `.safetensors` file a user already has on disk — which is the
case issue #3286 reports, where the same weights would otherwise be stored twice.

This module answers the question the Hub tags used to answer, from the file
itself.  It reads only the safetensors **header** — the JSON index of tensor
names and shapes at the front of the file — so identifying a 24 GB checkpoint
costs a few milliseconds and never touches the weights.

Two things make this more than a call to diffusers' own inference:

* `infer_diffusers_model_type` **never reports "unknown"**.  Its final branch
  returns ``"v1"``, so an unrecognised checkpoint is silently announced as Stable
  Diffusion 1.5.  We accept a ``v1`` answer only when the SD 1.5 marker key is
  actually present.
* Chroma is a FLUX derivative and carries FLUX's marker key, so diffusers
  identifies a Chroma checkpoint as ``flux-schnell``.  Families that diffusers
  cannot distinguish are matched here first, against signatures derived from the
  diffusers model classes themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Iterable, Literal

# Family ids as used in issue #3286.
FAMILY_SD15 = "sd-1.5"
FAMILY_SDXL = "sdxl"
FAMILY_FLUX = "flux"
FAMILY_FLUX2 = "flux2"
FAMILY_QWEN = "qwen"
FAMILY_Z_IMAGE = "z-image"
FAMILY_CHROMA = "chroma"

KNOWN_FAMILIES: tuple[str, ...] = (
    FAMILY_SD15,
    FAMILY_SDXL,
    FAMILY_FLUX,
    FAMILY_FLUX2,
    FAMILY_QWEN,
    FAMILY_Z_IMAGE,
    FAMILY_CHROMA,
)

# A checkpoint saved in the original (ComfyUI / reference) layout prefixes every
# tensor with this; the diffusers-native layout does not.
_ORIGINAL_PREFIX = "model.diffusion_model."

# Families diffusers' own inference cannot name, matched before we defer to it.
# Each signature is a set of keys that must all be present, derived from the
# state dict of the corresponding diffusers model class rather than guessed.
#
#   Chroma: replaces FLUX's `time_text_embed` with `distilled_guidance_layer`,
#           but keeps FLUX's `double_blocks` marker — so it must be matched
#           before FLUX or it loads as FLUX.
#   Qwen:   `txt_norm` appears in no other family here.
_NODETOOL_SIGNATURES: tuple[tuple[str, frozenset[str]], ...] = (
    (FAMILY_CHROMA, frozenset({"distilled_guidance_layer.in_proj.weight"})),
    (FAMILY_QWEN, frozenset({"txt_norm.weight"})),
)

# diffusers `infer_diffusers_model_type` result -> our family id.
_DIFFUSERS_MODEL_TYPES: dict[str, str] = {
    "v1": FAMILY_SD15,
    "xl_base": FAMILY_SDXL,
    "xl_refiner": FAMILY_SDXL,
    "flux-dev": FAMILY_FLUX,
    "flux-schnell": FAMILY_FLUX,
    "flux-2-dev": FAMILY_FLUX2,
    "z-image-turbo": FAMILY_Z_IMAGE,
}

# diffusers returns "v1" as its fallthrough rather than reporting failure, so a
# "v1" answer is only believed when this key is really in the checkpoint.
_SD15_MARKER = "model.diffusion_model.output_blocks.11.0.skip_connection.weight"


@dataclass(frozen=True)
class SingleFileLoadPlan:
    """How to turn one checkpoint file into a text-to-image pipeline.

    ``load_kind`` is the shape of the load, and it is a property of diffusers,
    not a preference:

    ``"pipeline"``
        The pipeline class implements ``from_single_file`` and reads the whole
        checkpoint itself.
    ``"transformer"``
        It does not.  The transformer is loaded from the file and the remaining
        components (text encoder, VAE, scheduler) come from ``base_repo_id``.
    """

    family: str
    pipeline_module: str
    pipeline_class: str
    load_kind: Literal["pipeline", "transformer"]
    dtype: Literal["float16", "bfloat16"]
    base_repo_id: str | None = None
    transformer_module: str | None = None
    transformer_class: str | None = None

    def __post_init__(self) -> None:
        if self.load_kind == "transformer" and not self.base_repo_id:
            raise ValueError(
                f"{self.family}: a transformer-only load needs a base repo for "
                "the text encoder, VAE and scheduler"
            )


_LOAD_PLANS: dict[str, SingleFileLoadPlan] = {
    FAMILY_SD15: SingleFileLoadPlan(
        family=FAMILY_SD15,
        pipeline_module="diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion",
        pipeline_class="StableDiffusionPipeline",
        load_kind="pipeline",
        dtype="float16",
    ),
    FAMILY_SDXL: SingleFileLoadPlan(
        family=FAMILY_SDXL,
        pipeline_module="diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl",
        pipeline_class="StableDiffusionXLPipeline",
        load_kind="pipeline",
        dtype="float16",
    ),
    FAMILY_FLUX: SingleFileLoadPlan(
        family=FAMILY_FLUX,
        pipeline_module="diffusers.pipelines.flux.pipeline_flux",
        pipeline_class="FluxPipeline",
        load_kind="pipeline",
        dtype="bfloat16",
    ),
    FAMILY_CHROMA: SingleFileLoadPlan(
        family=FAMILY_CHROMA,
        pipeline_module="diffusers.pipelines.chroma.pipeline_chroma",
        pipeline_class="ChromaPipeline",
        load_kind="pipeline",
        dtype="bfloat16",
    ),
    FAMILY_Z_IMAGE: SingleFileLoadPlan(
        family=FAMILY_Z_IMAGE,
        pipeline_module="diffusers.pipelines.z_image.pipeline_z_image",
        pipeline_class="ZImagePipeline",
        load_kind="pipeline",
        dtype="bfloat16",
    ),
    # QwenImagePipeline and Flux2Pipeline have no `from_single_file`; their
    # transformers do.  The base repos are the ones this package already names
    # for those families elsewhere.
    FAMILY_QWEN: SingleFileLoadPlan(
        family=FAMILY_QWEN,
        pipeline_module="diffusers.pipelines.qwenimage.pipeline_qwenimage",
        pipeline_class="QwenImagePipeline",
        load_kind="transformer",
        dtype="bfloat16",
        base_repo_id="Qwen/Qwen-Image",
        transformer_module="diffusers.models.transformers.transformer_qwenimage",
        transformer_class="QwenImageTransformer2DModel",
    ),
    FAMILY_FLUX2: SingleFileLoadPlan(
        family=FAMILY_FLUX2,
        pipeline_module="diffusers.pipelines.flux2.pipeline_flux2",
        pipeline_class="Flux2Pipeline",
        load_kind="transformer",
        dtype="bfloat16",
        base_repo_id="black-forest-labs/FLUX.2-dev",
        transformer_module="diffusers.models.transformers.transformer_flux2",
        transformer_class="Flux2Transformer2DModel",
    ),
}


class UnknownCheckpointFamily(ValueError):
    """Raised when a checkpoint matches no family NodeTool can load."""


def _strip_prefix(key: str) -> str:
    return key[len(_ORIGINAL_PREFIX) :] if key.startswith(_ORIGINAL_PREFIX) else key


def detect_family(keys: Iterable[str] | Mapping[str, tuple[int, ...]]) -> str | None:
    """Name the model family of a checkpoint from its tensor names and shapes.

    Accepts a bare iterable of key names, or the name -> shape mapping
    ``read_checkpoint_shapes`` returns.  A few of diffusers' detection branches
    read tensor shapes, so pass the mapping when you have it.

    Returns ``None`` when no family matches, which diffusers' own inference will
    not do.
    """
    shapes: dict[str, tuple[int, ...]]
    if isinstance(keys, Mapping):
        shapes = dict(keys)
    else:
        shapes = {k: () for k in keys}
    if not shapes:
        return None
    # Match against both layouts without duplicating every signature.
    unprefixed = {_strip_prefix(k) for k in shapes}

    for family, signature in _NODETOOL_SIGNATURES:
        if signature <= unprefixed:
            return family

    model_type = _infer_with_diffusers(shapes)
    if model_type is None:
        return None
    if model_type == "v1" and _SD15_MARKER not in shapes:
        # diffusers fell through to its default rather than recognising SD 1.5.
        return None
    return _DIFFUSERS_MODEL_TYPES.get(model_type)


def _infer_with_diffusers(shapes: dict[str, tuple[int, ...]]) -> str | None:
    from diffusers.loaders.single_file_utils import infer_diffusers_model_type

    try:
        return infer_diffusers_model_type(_HeaderCheckpoint(shapes))
    except Exception:
        # A detection branch needed something the header does not carry.  Saying
        # nothing is better than guessing.
        return None


class _HeaderCheckpoint(dict):
    """A checkpoint stand-in whose values expose only a shape.

    ``infer_diffusers_model_type`` reads key membership and, for a few families,
    ``checkpoint[key].shape``.  Both come out of the safetensors header, so no
    weights are read.
    """

    def __init__(self, shapes: Mapping[str, tuple[int, ...]]):
        super().__init__({k: _ShapeOnly(tuple(v)) for k, v in shapes.items()})


@dataclass(frozen=True)
class _ShapeOnly:
    shape: tuple[int, ...]


def read_checkpoint_shapes(path: str) -> dict[str, tuple[int, ...]]:
    """Read tensor names and shapes from a safetensors header. No weights."""
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        return {k: tuple(handle.get_slice(k).get_shape()) for k in handle.keys()}


def plan_for_family(family: str) -> SingleFileLoadPlan:
    try:
        return _LOAD_PLANS[family]
    except KeyError:
        raise UnknownCheckpointFamily(f"No load plan for family {family!r}") from None


def detect_single_file_checkpoint(path: str) -> SingleFileLoadPlan:
    """Identify a checkpoint on disk and say how to load it.

    Raises ``UnknownCheckpointFamily`` naming the families NodeTool does support,
    rather than loading the file as something it is not.
    """
    shapes = read_checkpoint_shapes(path)
    family = detect_family(shapes.keys())
    if family is None:
        raise UnknownCheckpointFamily(
            f"Could not identify the model family of {path}. NodeTool reads "
            f"single-file checkpoints for: {', '.join(KNOWN_FAMILIES)}."
        )
    return plan_for_family(family)


def resolve_local_checkpoint(model_id: str | None, model_path: str | None) -> str | None:
    """Return the checkpoint file a model reference names on disk, or ``None``.

    NodeTool's model reference is a Hub repo id plus a filename inside it.  A
    reference whose ``path`` — or, when there is no separate path, whose
    ``repo_id`` — resolves to an existing file is a local checkpoint instead, and
    nothing about it is fetched from the Hub.

    A bare filename is deliberately not treated as local: it would make the
    meaning of a model reference depend on the process's working directory.
    """
    import os

    for candidate in (model_path, model_id):
        if not candidate:
            continue
        expanded = os.path.expanduser(candidate)
        if os.path.isabs(expanded) and os.path.isfile(expanded):
            return expanded
    return None
