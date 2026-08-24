"""Identifying a single-file checkpoint on disk (issue nodetool-ai/nodetool#3286).

Every checkpoint here is synthetic: a real safetensors file carrying the marker
tensors of a family and nothing else, a few hundred bytes each.  Detection reads
the header, so a file with one 4-element tensor exercises exactly the code path a
24 GB checkpoint does.  No network, no weights, no GPU.
"""

import pytest
import torch
from safetensors.torch import save_file

from nodetool.huggingface.single_file_models import (
    FAMILY_CHROMA,
    FAMILY_FLUX,
    FAMILY_FLUX2,
    FAMILY_QWEN,
    FAMILY_SD15,
    FAMILY_SDXL,
    FAMILY_Z_IMAGE,
    KNOWN_FAMILIES,
    SingleFileLoadPlan,
    UnknownCheckpointFamily,
    detect_family,
    detect_single_file_checkpoint,
    plan_for_family,
    read_checkpoint_shapes,
    resolve_local_checkpoint,
)

# The marker tensors that identify each family. The SD 1.5, SDXL, FLUX, FLUX.2
# and Z-Image markers are diffusers' own (CHECKPOINT_KEY_NAMES); the Qwen and
# Chroma ones are taken from those models' diffusers classes, which diffusers'
# inference does not cover.
FAMILY_MARKERS: dict[str, list[str]] = {
    FAMILY_SD15: ["model.diffusion_model.output_blocks.11.0.skip_connection.weight"],
    FAMILY_SDXL: [
        "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias"
    ],
    FAMILY_FLUX: ["double_blocks.0.img_attn.norm.key_norm.scale"],
    FAMILY_FLUX2: ["model.diffusion_model.single_stream_modulation.lin.weight"],
    FAMILY_Z_IMAGE: ["model.diffusion_model.layers.0.adaLN_modulation.0.weight"],
    FAMILY_QWEN: ["txt_norm.weight", "transformer_blocks.0.img_mod.1.weight"],
    # Chroma is a FLUX derivative: it carries FLUX's marker too. That overlap is
    # the point — see test_chroma_is_not_mistaken_for_flux.
    FAMILY_CHROMA: [
        "double_blocks.0.img_attn.norm.key_norm.scale",
        "distilled_guidance_layer.in_proj.weight",
    ],
}


def write_checkpoint(path, keys) -> str:
    save_file({key: torch.zeros(4) for key in keys}, str(path))
    return str(path)


@pytest.mark.parametrize("family", sorted(FAMILY_MARKERS))
def test_each_requested_family_is_detected(tmp_path, family):
    path = write_checkpoint(tmp_path / f"{family}.safetensors", FAMILY_MARKERS[family])
    assert detect_family(read_checkpoint_shapes(path)) == family


def test_every_requested_family_has_a_marker_and_a_plan():
    """The families the issue asked for, all seven, are covered."""
    assert set(FAMILY_MARKERS) == set(KNOWN_FAMILIES)
    for family in KNOWN_FAMILIES:
        assert isinstance(plan_for_family(family), SingleFileLoadPlan)


def test_chroma_is_not_mistaken_for_flux():
    """Chroma carries FLUX's marker key, so order of matching decides this."""
    from diffusers.loaders.single_file_utils import infer_diffusers_model_type

    chroma_keys = FAMILY_MARKERS[FAMILY_CHROMA]

    # diffusers on its own calls this FLUX — the premise for matching Chroma first.
    class _Shape:
        shape = ()

    assert infer_diffusers_model_type({k: _Shape() for k in chroma_keys}).startswith(
        "flux"
    )
    assert detect_family(chroma_keys) == FAMILY_CHROMA


def test_unrecognised_checkpoint_is_not_called_sd15():
    """diffusers falls through to "v1"; we must report not-knowing instead.

    Without this, any unknown checkpoint loads as Stable Diffusion 1.5 and fails
    somewhere deep in the weight conversion, or worse, quietly produces noise.
    """
    from diffusers.loaders.single_file_utils import infer_diffusers_model_type

    class _Shape:
        shape = ()

    junk = {"totally.unrelated.tensor": _Shape()}
    assert infer_diffusers_model_type(junk) == "v1"
    assert detect_family(["totally.unrelated.tensor"]) is None


def test_unknown_file_raises_and_names_the_supported_families(tmp_path):
    path = write_checkpoint(tmp_path / "junk.safetensors", ["totally.unrelated.tensor"])
    with pytest.raises(UnknownCheckpointFamily) as excinfo:
        detect_single_file_checkpoint(path)
    for family in KNOWN_FAMILIES:
        assert family in str(excinfo.value)


def test_empty_checkpoint_is_unknown():
    assert detect_family([]) is None


def test_detect_returns_a_usable_plan(tmp_path):
    path = write_checkpoint(tmp_path / "sdxl.safetensors", FAMILY_MARKERS[FAMILY_SDXL])
    plan = detect_single_file_checkpoint(path)
    assert plan.family == FAMILY_SDXL
    assert plan.load_kind == "pipeline"


def test_reading_a_header_does_not_read_weights(tmp_path):
    """Detection is header-only, so it must not depend on tensor contents."""
    keys = FAMILY_MARKERS[FAMILY_FLUX]
    path = str(tmp_path / "flux.safetensors")
    save_file({k: torch.full((4,), 3.5) for k in keys}, path)
    shapes = read_checkpoint_shapes(path)
    assert shapes == {keys[0]: (4,)}


@pytest.mark.parametrize(
    "family, load_kind",
    [
        (FAMILY_SD15, "pipeline"),
        (FAMILY_SDXL, "pipeline"),
        (FAMILY_FLUX, "pipeline"),
        (FAMILY_CHROMA, "pipeline"),
        (FAMILY_Z_IMAGE, "pipeline"),
        (FAMILY_QWEN, "transformer"),
        (FAMILY_FLUX2, "transformer"),
    ],
)
def test_load_kind_matches_what_diffusers_actually_supports(family, load_kind):
    """`load_kind` is a fact about diffusers, not a preference.

    A pipeline class that grows `from_single_file` upstream should flip to the
    simpler load here, and this test is what says so.
    """
    import importlib

    plan = plan_for_family(family)
    assert plan.load_kind == load_kind
    pipeline_cls = getattr(
        importlib.import_module(plan.pipeline_module), plan.pipeline_class
    )
    assert hasattr(pipeline_cls, "from_single_file") == (load_kind == "pipeline")

    if load_kind == "transformer":
        assert plan.base_repo_id
        transformer_cls = getattr(
            importlib.import_module(plan.transformer_module), plan.transformer_class
        )
        assert hasattr(transformer_cls, "from_single_file")


def test_transformer_plan_without_a_base_repo_is_rejected():
    with pytest.raises(ValueError, match="base repo"):
        SingleFileLoadPlan(
            family="made-up",
            pipeline_module="m",
            pipeline_class="C",
            load_kind="transformer",
            dtype="bfloat16",
        )


class TestResolveLocalCheckpoint:
    def test_absolute_path_in_path_field(self, tmp_path):
        path = write_checkpoint(tmp_path / "m.safetensors", ["a.b"])
        assert resolve_local_checkpoint("some/repo", path) == path

    def test_absolute_path_in_repo_id_field(self, tmp_path):
        path = write_checkpoint(tmp_path / "m.safetensors", ["a.b"])
        assert resolve_local_checkpoint(path, None) == path

    def test_hub_reference_stays_a_hub_reference(self):
        assert (
            resolve_local_checkpoint("stabilityai/sdxl", "model.safetensors") is None
        )

    def test_absolute_path_that_does_not_exist_is_not_local(self):
        assert resolve_local_checkpoint(None, "/no/such/checkpoint.safetensors") is None

    def test_bare_filename_is_never_local(self, tmp_path, monkeypatch):
        """A relative name must not depend on the working directory."""
        write_checkpoint(tmp_path / "m.safetensors", ["a.b"])
        monkeypatch.chdir(tmp_path)
        assert resolve_local_checkpoint("repo", "m.safetensors") is None


class TestPipelineRouting:
    """A local checkpoint must bypass the Hub entirely.

    Today's single-file path calls `fetch_model_info` to read the repo's tags and
    raises when they are missing, which is exactly why a file on disk could not
    be loaded at all.
    """

    @pytest.fixture
    def context(self):
        class _Context:
            device = "cpu"

        return _Context()

    @pytest.mark.asyncio
    async def test_local_checkpoint_never_calls_the_hub(
        self, tmp_path, monkeypatch, context
    ):
        from nodetool.huggingface import text_to_image_pipelines as mod

        path = write_checkpoint(
            tmp_path / "sdxl.safetensors", FAMILY_MARKERS[FAMILY_SDXL]
        )

        async def fail(*args, **kwargs):
            raise AssertionError("the Hub was contacted for a local checkpoint")

        monkeypatch.setattr(mod, "fetch_model_info", fail)
        monkeypatch.setattr(mod, "_ensure_file_cached", fail)

        built = {}

        async def fake_build(checkpoint_path, *, plan=None):
            built["path"] = checkpoint_path
            built["family"] = plan.family
            return object()

        monkeypatch.setattr(mod, "load_local_single_file_pipeline", fake_build)
        monkeypatch.setattr(mod, "_apply_memory_optimizations", lambda *a, **k: False)
        monkeypatch.setattr(mod.ModelManager, "get_model", lambda *a, **k: None)
        monkeypatch.setattr(mod.ModelManager, "set_model", lambda *a, **k: None)

        pipeline, _ = await mod.load_text_to_image_pipeline(
            context=context, model_id=path, model_path=None, node_id="n1"
        )

        assert pipeline is not None
        assert built == {"path": path, "family": FAMILY_SDXL}

    @pytest.mark.asyncio
    async def test_hub_reference_still_takes_the_hub_path(
        self, monkeypatch, context
    ):
        """The local route must not swallow ordinary repo-id models."""
        from nodetool.huggingface import text_to_image_pipelines as mod

        async def fail_local(*args, **kwargs):
            raise AssertionError("a Hub model took the local-file route")

        monkeypatch.setattr(mod, "load_local_single_file_pipeline", fail_local)
        monkeypatch.setattr(mod.ModelManager, "get_model", lambda *a, **k: None)

        reached = {}

        async def fake_fetch(model_id):
            reached["model_id"] = model_id
            raise RuntimeError("stop here")

        monkeypatch.setattr(mod, "fetch_model_info", fake_fetch)
        monkeypatch.setattr(mod.HF_FAST_CACHE, "resolve", fake_resolve_ok)

        with pytest.raises(RuntimeError, match="stop here"):
            await mod.load_text_to_image_pipeline(
                context=context,
                model_id="stabilityai/sdxl",
                model_path="model.safetensors",
                node_id="n1",
            )
        assert reached["model_id"] == "stabilityai/sdxl"


async def fake_resolve_ok(*args, **kwargs):
    return "/tmp/cached.safetensors"
