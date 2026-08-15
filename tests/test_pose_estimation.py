import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from nodetool.nodes.huggingface.pose_estimation import (
    COCO_KEYPOINT_LABELS,
    COCO_SKELETON,
    PoseEstimation,
    VisualizePoseEstimation,
    is_moe_model,
    person_label_ids,
    pose_result_to_dict,
    to_list,
    xyxy_to_xywh,
)


class FakeTensor:
    """Minimal stand-in for a torch tensor / numpy array."""

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class FakeConfig:
    def __init__(self, id2label=None, num_experts=None):
        if id2label is not None:
            self.id2label = id2label
        if num_experts is not None:
            self.backbone_config = FakeBackboneConfig(num_experts)


class FakeBackboneConfig:
    def __init__(self, num_experts):
        self.num_experts = num_experts


class FakeModel:
    def __init__(self, config):
        self.config = config


# --- node metadata -----------------------------------------------------------


def test_pose_estimation_default_models():
    node = PoseEstimation()
    assert node.model.repo_id == "usyd-community/vitpose-base-simple"
    assert node.person_detector.repo_id == "PekingU/rtdetr_r50vd_coco_o365"


def test_pose_estimation_recommended_models():
    repo_ids = [m.repo_id for m in PoseEstimation.get_recommended_models()]
    assert "usyd-community/vitpose-base-simple" in repo_ids
    assert "usyd-community/vitpose-plus-huge" in repo_ids
    assert all(r.startswith("usyd-community/vitpose") for r in repo_ids)


def test_pose_estimation_basic_fields_and_required_inputs():
    assert PoseEstimation.get_basic_fields() == ["model", "image"]
    assert PoseEstimation.required_inputs(PoseEstimation()) == ["image"]


def test_pose_estimation_title():
    assert PoseEstimation.get_title() == "HF Pose Estimation"


def test_visualize_title_and_required_inputs():
    assert VisualizePoseEstimation.get_title() == "Visualize Pose Estimation"
    assert VisualizePoseEstimation.required_inputs(VisualizePoseEstimation()) == [
        "image",
        "poses",
    ]


def test_model_ids_read_from_fields():
    node = PoseEstimation()
    assert node.get_model_id() == node.model.repo_id
    assert node.get_detector_model_id() == node.person_detector.repo_id


# --- skeleton constants ------------------------------------------------------


def test_coco_keypoint_labels():
    assert len(COCO_KEYPOINT_LABELS) == 17
    assert COCO_KEYPOINT_LABELS[0] == "nose"
    assert COCO_KEYPOINT_LABELS[-1] == "right_ankle"


def test_coco_skeleton_indices_in_range():
    assert COCO_SKELETON
    for start, end in COCO_SKELETON:
        assert 0 <= start < 17
        assert 0 <= end < 17
        assert start != end


# --- helpers -----------------------------------------------------------------


def test_to_list_handles_tensors_lists_and_scalars():
    assert to_list(FakeTensor([[1.0, 2.0]])) == [[1.0, 2.0]]
    assert to_list([FakeTensor([1]), FakeTensor([2])]) == [[1], [2]]
    assert to_list(None) is None
    assert to_list(3) == 3


def test_xyxy_to_xywh_conversion():
    assert xyxy_to_xywh([[10, 20, 40, 80]]) == [[10.0, 20.0, 30.0, 60.0]]


def test_xyxy_to_xywh_ignores_extra_columns():
    assert xyxy_to_xywh([[0, 0, 5, 5, 0.9]]) == [[0.0, 0.0, 5.0, 5.0]]


def test_person_label_ids_from_id2label():
    config = FakeConfig(id2label={0: "cat", 1: "person", 2: "dog"})
    assert person_label_ids(config) == {1}


def test_person_label_ids_defaults_to_zero():
    assert person_label_ids(FakeConfig()) == {0}
    assert person_label_ids(FakeConfig(id2label={0: "cat"})) == {0}


def test_person_label_ids_with_string_keys():
    config = FakeConfig(id2label={"0": "cat", "3": "Person"})
    assert person_label_ids(config) == {3}


def test_is_moe_model_by_name():
    assert is_moe_model(FakeModel(FakeConfig()), "usyd-community/vitpose-plus-base")


def test_is_moe_model_by_config():
    model = FakeModel(FakeConfig(num_experts=6))
    assert is_moe_model(model, "some/model")


def test_is_moe_model_false_for_simple():
    model = FakeModel(FakeConfig(num_experts=1))
    assert not is_moe_model(model, "usyd-community/vitpose-base-simple")
    assert not is_moe_model(FakeModel(FakeConfig()), "usyd-community/vitpose-base")


# --- result conversion -------------------------------------------------------


def _fake_pose_result():
    return {
        "keypoints": FakeTensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
        "scores": FakeTensor([0.9, 0.1, 0.8]),
        "labels": FakeTensor([0, 1, 2]),
        "bbox": FakeTensor([5.0, 6.0, 100.0, 200.0]),
    }


def test_pose_result_to_dict_structure():
    pose = pose_result_to_dict(_fake_pose_result(), keypoint_threshold=0.0)
    assert [kp["label"] for kp in pose["keypoints"]] == [
        "nose",
        "left_eye",
        "right_eye",
    ]
    first = pose["keypoints"][0]
    assert first == {
        "index": 0,
        "label": "nose",
        "x": 10.0,
        "y": 20.0,
        "score": 0.9,
    }
    assert pose["bbox"] == {"x": 5.0, "y": 6.0, "width": 100.0, "height": 200.0}
    assert pose["score"] == pytest.approx((0.9 + 0.1 + 0.8) / 3)


def test_pose_result_to_dict_applies_keypoint_threshold():
    pose = pose_result_to_dict(_fake_pose_result(), keypoint_threshold=0.5)
    assert [kp["index"] for kp in pose["keypoints"]] == [0, 2]


def test_pose_result_to_dict_without_labels_uses_position():
    result = {
        "keypoints": [[1.0, 2.0], [3.0, 4.0]],
        "scores": [1.0, 1.0],
    }
    pose = pose_result_to_dict(result)
    assert [kp["label"] for kp in pose["keypoints"]] == ["nose", "left_eye"]
    assert "bbox" not in pose


def test_pose_result_to_dict_empty():
    assert pose_result_to_dict({}) == {"keypoints": []}


def test_pose_result_to_dict_unknown_label_index():
    result = {
        "keypoints": [[1.0, 2.0]],
        "scores": [1.0],
        "labels": [99],
    }
    pose = pose_result_to_dict(result)
    assert pose["keypoints"][0]["label"] == "99"
