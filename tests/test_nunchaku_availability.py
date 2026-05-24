from __future__ import annotations

import sys
from types import ModuleType

import pytest

from nodetool.huggingface import nunchaku_pipelines


@pytest.fixture(autouse=True)
def reset_nunchaku_cache():
    nunchaku_pipelines.NUNCHAKU_AVAILABLE = None
    yield
    nunchaku_pipelines.NUNCHAKU_AVAILABLE = None


def test_check_nunchaku_available_rejects_unrelated_pypi_package(monkeypatch):
    fake = ModuleType("nunchaku")

    monkeypatch.setitem(sys.modules, "nunchaku", fake)

    assert nunchaku_pipelines._check_nunchaku_available() is False


def test_check_nunchaku_available_accepts_svdquant_runtime(monkeypatch):
    fake = ModuleType("nunchaku")
    fake.NunchakuFluxTransformer2dModel = object()
    monkeypatch.setitem(sys.modules, "nunchaku", fake)

    assert nunchaku_pipelines._check_nunchaku_available() is True
