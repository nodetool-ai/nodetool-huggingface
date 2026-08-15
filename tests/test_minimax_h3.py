import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.metadata.types import AudioRef, ImageRef, VideoRef
from nodetool.nodes.huggingface.text_to_video import (
    MINIMAX_H3_MODULE,
    PX_FRAMES_OOM,
    PX_FRAMES_PROVEN,
    MiniMaxH3,
    MiniMaxH3Reference,
    _MiniMaxH3Base,
    _load_minimax_h3_runtime,
)
from nodetool.nodes.huggingface import text_to_video


def test_minimax_h3_titles():
    assert MiniMaxH3.get_title() == "MiniMax-H3"
    assert MiniMaxH3Reference.get_title() == "MiniMax-H3 Reference"


def test_base_node_is_hidden():
    assert not _MiniMaxH3Base.is_visible()
    assert MiniMaxH3.is_visible()
    assert MiniMaxH3Reference.is_visible()


def test_workflows_select_the_right_checkpoint_partition():
    # `fl2va` loads `transformer/` and also serves text-only requests;
    # `ref2va` loads `transformer_ref/`.
    assert MiniMaxH3()._workflow == "fl2va"
    assert MiniMaxH3Reference()._workflow == "ref2va"


def test_recommended_models():
    for node_class in (MiniMaxH3, MiniMaxH3Reference):
        models = node_class.get_recommended_models()
        assert [m.repo_id for m in models] == ["MiniMaxAI/MiniMax-H3"]


def test_model_id():
    assert MiniMaxH3().get_model_id() == "MiniMaxAI/MiniMax-H3"


def test_frame_count_bounds_match_the_five_to_fifteen_second_window():
    field = MiniMaxH3.model_fields["num_frames"]
    metadata = {type(m).__name__: m for m in field.metadata}
    # 120 frames is 5s at 24fps; 345 is the largest 17n+5 within 15s.
    assert metadata["Ge"].ge == 120
    assert metadata["Le"].le == 345
    assert field.default == 124


def test_canvas_is_left_to_the_model_unless_both_axes_are_set():
    assert MiniMaxH3()._canvas_kwargs() == {}
    assert MiniMaxH3(height=544)._canvas_kwargs() == {}
    assert MiniMaxH3(height=544, width=960)._canvas_kwargs() == {
        "height": 544,
        "width": 960,
    }


def test_off_lattice_requests_are_snapped_before_the_pipeline_sees_them():
    # 130 frames is off the 17n+5 lattice and 1200x700 is off the 32px grid.
    assert MiniMaxH3(num_frames=130, width=1200, height=700)._geometry() == (
        1216,
        704,
        124,
    )
    assert MiniMaxH3(num_frames=130)._geometry() == (0, 0, 124)
    assert MiniMaxH3(num_frames=130, width=1200, height=700)._canvas_kwargs() == {
        "height": 704,
        "width": 1216,
    }


def test_the_proven_pixel_frame_budget_is_the_measured_one():
    # Measured on a 24GB RTX 4090: 1216x704x209 fits, 1216x704x226 OOMs.
    assert 1216 * 704 * 209 <= PX_FRAMES_PROVEN
    assert 1216 * 704 * 226 >= PX_FRAMES_OOM


def test_a_request_over_the_joint_budget_is_refused_without_loading_anything():
    node = MiniMaxH3(width=1216, height=704, num_frames=226)
    with pytest.raises(ValueError, match="pixel-frames") as excinfo:
        node._check_pixel_budget()

    message = str(excinfo.value)
    assert "179M" in message
    assert "num_frames" in message
    assert node._pipeline is None


def test_a_request_inside_the_joint_budget_passes():
    MiniMaxH3(width=1216, height=704, num_frames=209)._check_pixel_budget()
    # Each axis on its own is legal here; only the product is not.
    with pytest.raises(ValueError, match="pixel-frames"):
        MiniMaxH3(width=1344, height=1344, num_frames=345)._check_pixel_budget()


def test_the_budget_check_is_skipped_when_the_model_picks_the_canvas():
    MiniMaxH3(num_frames=345)._check_pixel_budget()


def test_the_budget_check_runs_before_the_pipeline_is_loaded(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("the H3 runtime must not be loaded for a refused request")

    monkeypatch.setattr(text_to_video, "_load_minimax_h3_runtime", fail)

    node = MiniMaxH3(width=1216, height=704, num_frames=226)
    with pytest.raises(ValueError, match="pixel-frames"):
        import asyncio

        asyncio.run(node.preload_model(None))


def test_reference_node_also_enforces_the_budget():
    with pytest.raises(ValueError, match="pixel-frames"):
        MiniMaxH3Reference(width=1216, height=704, num_frames=226)._check_pixel_budget()


def test_no_guidance_fields_on_a_distilled_checkpoint():
    for node_class in (MiniMaxH3, MiniMaxH3Reference):
        assert "negative_prompt" not in node_class.model_fields
        assert "guidance_scale" not in node_class.model_fields


def test_outputs_carry_both_video_and_audio():
    for node_class in (MiniMaxH3, MiniMaxH3Reference):
        outputs = node_class.get_metadata(include_model_info=False).outputs
        assert [(slot.name, slot.type.type) for slot in outputs] == [
            ("video", "video"),
            ("audio", "audio"),
        ]


def test_reference_limits():
    node = MiniMaxH3Reference(image_references=[ImageRef()] * 10)
    with pytest.raises(ValueError, match="9 image references"):
        node._validate_references()

    node = MiniMaxH3Reference(video_references=[VideoRef()] * 4)
    with pytest.raises(ValueError, match="3 video references"):
        node._validate_references()

    node = MiniMaxH3Reference(audio_references=[AudioRef()] * 4)
    with pytest.raises(ValueError, match="3 audio references"):
        node._validate_references()

    node = MiniMaxH3Reference(
        image_references=[ImageRef()] * 9,
        video_references=[VideoRef()] * 3,
        audio_references=[AudioRef()] * 1,
    )
    with pytest.raises(ValueError, match="12 references"):
        node._validate_references()


def test_audio_references_need_a_visual_reference():
    node = MiniMaxH3Reference(audio_references=[AudioRef()])
    with pytest.raises(ValueError, match="image or video reference"):
        node._validate_references()


def test_at_least_one_reference_is_required():
    with pytest.raises(ValueError, match="at least one reference"):
        MiniMaxH3Reference()._validate_references()


def test_a_valid_reference_mix_passes():
    node = MiniMaxH3Reference(
        image_references=[ImageRef()] * 9,
        video_references=[VideoRef()] * 3,
    )
    node._validate_references()


def test_h3_runtime_is_loaded_dynamically(monkeypatch):
    integration = object()
    components_manager = object()
    modular_pipeline = object()
    diffusers = SimpleNamespace(
        __version__="0.40.0",
        ComponentsManager=components_manager,
        ModularPipeline=modular_pipeline,
    )

    def import_module(name):
        return integration if name == MINIMAX_H3_MODULE else diffusers

    monkeypatch.setattr(text_to_video.importlib, "import_module", import_module)

    assert _load_minimax_h3_runtime() == (
        integration,
        components_manager,
        modular_pipeline,
    )


def test_diffusers_039_keeps_nodes_importable_and_reports_h3_unavailable(monkeypatch):
    def import_module(name):
        if name == MINIMAX_H3_MODULE:
            raise ModuleNotFoundError(name=MINIMAX_H3_MODULE)
        return SimpleNamespace(__version__="0.39.0")

    monkeypatch.setattr(text_to_video.importlib, "import_module", import_module)

    # Class construction and metadata do not import the H3 integration.
    assert MiniMaxH3.get_title() == "MiniMax-H3"
    assert MiniMaxH3Reference.get_title() == "MiniMax-H3 Reference"

    with pytest.raises(ImportError, match=r"Diffusers 0\.39.*does not include"):
        _load_minimax_h3_runtime()
