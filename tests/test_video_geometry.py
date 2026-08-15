import logging

import pytest

from nodetool.nodes.huggingface.text_to_video import (
    LTX2_GEOMETRY,
    MINIMAX_H3_GEOMETRY,
    VideoGeometry,
    snap_video_dimension,
    snap_video_frames,
    snap_video_geometry,
)


def test_legal_geometry_passes_through_untouched(caplog):
    with caplog.at_level(logging.WARNING):
        assert snap_video_geometry(768, 512, 121, LTX2_GEOMETRY, "LTX-2") == (
            768,
            512,
            121,
        )
    assert caplog.records == []


def test_off_lattice_frames_snap_to_the_nearest_legal_count():
    assert snap_video_frames(120, LTX2_GEOMETRY, "LTX-2") == 121
    assert snap_video_frames(124, LTX2_GEOMETRY, "LTX-2") == 121
    assert snap_video_frames(126, LTX2_GEOMETRY, "LTX-2") == 129
    # 17n+5 with a floor that is not itself on the lattice.
    assert snap_video_frames(120, MINIMAX_H3_GEOMETRY, "MiniMax-H3") == 124
    assert snap_video_frames(200, MINIMAX_H3_GEOMETRY, "MiniMax-H3") == 192
    assert snap_video_frames(205, MINIMAX_H3_GEOMETRY, "MiniMax-H3") == 209


def test_snapping_stays_inside_the_supported_range():
    assert snap_video_frames(257, LTX2_GEOMETRY, "LTX-2") == 257
    assert snap_video_frames(9, LTX2_GEOMETRY, "LTX-2") == 9
    assert snap_video_frames(345, MINIMAX_H3_GEOMETRY, "MiniMax-H3") == 345


def test_off_multiple_dimensions_snap():
    assert snap_video_dimension(770, LTX2_GEOMETRY, "LTX-2", "width") == 768
    assert snap_video_dimension(500, LTX2_GEOMETRY, "LTX-2", "height") == 512
    assert (
        snap_video_dimension(1216, MINIMAX_H3_GEOMETRY, "MiniMax-H3", "width") == 1216
    )


def test_snapping_warns_when_it_changes_user_input(caplog):
    with caplog.at_level(logging.WARNING):
        snap_video_geometry(770, 500, 120, LTX2_GEOMETRY, "LTX-2")

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 3
    assert any("multiple of 32" in message and "768" in message for message in messages)
    assert any("8n+1" in message and "121" in message for message in messages)


def test_out_of_range_frames_raise():
    with pytest.raises(ValueError, match="9 to 257 frames"):
        snap_video_frames(4, LTX2_GEOMETRY, "LTX-2")
    with pytest.raises(ValueError, match="9 to 257 frames"):
        snap_video_frames(400, LTX2_GEOMETRY, "LTX-2")


def test_out_of_range_dimensions_raise():
    with pytest.raises(ValueError, match="width between 256 and 1280"):
        snap_video_dimension(64, LTX2_GEOMETRY, "LTX-2", "width")
    with pytest.raises(ValueError, match="height between 256 and 1280"):
        snap_video_dimension(2048, LTX2_GEOMETRY, "LTX-2", "height")


def test_a_lattice_whose_bounds_are_not_lattice_points_clamps_inward():
    spec = VideoGeometry(
        frame_step=8,
        frame_base=1,
        min_frames=10,
        max_frames=30,
        dim_multiple=32,
        min_dim=100,
        max_dim=200,
    )
    # 10 rounds to 9, which is below the floor, so it steps up to 17.
    assert snap_video_frames(10, spec, "test") == 17
    # 30 rounds to 33, which is above the ceiling, so it steps down to 25.
    assert snap_video_frames(30, spec, "test") == 25
    assert snap_video_dimension(100, spec, "test", "width") == 128
    assert snap_video_dimension(200, spec, "test", "width") == 192
