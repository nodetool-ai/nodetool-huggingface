"""Lightweight readiness facts for local TTS adapters."""

from dataclasses import dataclass
from importlib.util import find_spec

from nodetool.metadata.types import ModelAdapterInfo, ModelArtifactRef


@dataclass(frozen=True)
class TTSAdapterSpec:
    modules: tuple[str, ...]
    install_hint: str


_EXACT_SPECS: dict[str, TTSAdapterSpec] = {
    "hexgrad/Kokoro-82M": TTSAdapterSpec(
        modules=("kokoro",),
        install_hint="pip install nodetool-huggingface",
    ),
    "Supertone/supertonic-3": TTSAdapterSpec(
        modules=("supertonic",),
        install_hint="pip install nodetool-huggingface[supertonic]",
    ),
    "SWivid/F5-TTS": TTSAdapterSpec(
        modules=("f5_tts", "torchcodec"),
        install_hint="pip install nodetool-huggingface[f5-tts]",
    ),
    "suno/bark": TTSAdapterSpec(
        modules=("transformers",),
        install_hint="pip install nodetool-huggingface",
    ),
    "suno/bark-small": TTSAdapterSpec(
        modules=("transformers",),
        install_hint="pip install nodetool-huggingface",
    ),
}

_TRANSFORMERS_SPEC = TTSAdapterSpec(
    modules=("transformers",),
    install_hint="pip install nodetool-huggingface",
)


def _spec_for(repo_id: str) -> TTSAdapterSpec | None:
    if repo_id.startswith("facebook/mms-tts-"):
        return _TRANSFORMERS_SPEC
    return _EXACT_SPECS.get(repo_id)


def get_tts_adapter_info(repo_id: str) -> ModelAdapterInfo:
    """Return adapter readiness without importing a model runtime or weights."""

    artifact_ref = ModelArtifactRef(source="huggingface", repo_id=repo_id)
    spec = _spec_for(repo_id)
    if spec is None:
        return ModelAdapterInfo(state="unknown", artifact_ref=artifact_ref)

    missing = [module for module in spec.modules if find_spec(module) is None]
    if not missing:
        return ModelAdapterInfo(state="installed", artifact_ref=artifact_ref)

    names = ", ".join(missing)
    return ModelAdapterInfo(
        state="missing_dependency",
        reason_code="missing_dependency",
        reason=f"Missing Python package: {names}. Install with `{spec.install_hint}`.",
        artifact_ref=artifact_ref,
    )
