# Local TTS Expansion Plan — Reduced Scope

Research date: 2026-08-18

Implementation branch: `feat/expand-tts-support`

Primary repository: `nodetool-huggingface`

Related repositories: `nodetool-core` and `nodetool`

Source list: [Hugging Face text-to-speech models, trending](https://huggingface.co/models?pipeline_tag=text-to-speech&sort=trending)

## Scope

Add only low-risk local TTS models that have one of these integration paths:

- a maintained PyPI package compatible with NodeTool, or
- a standard/reviewable Hugging Face `AutoModel` interface, or
- one shared adapter that supports more than one checkpoint in the same family.

Do not add models that require a source checkout, an isolated dependency environment, a separate serving stack, a new native backend, or a one-off integration with no reuse.

Voice cloning remains required for included models that support it. A cloning model is complete only when reference audio reaches both its dedicated node and the generic `nodetool.audio.TextToSpeech` path.

## Completed research

- [x] Created a feature branch before changing the repository.
- [x] Inventoried existing Python, TypeScript, local, and provider-backed TTS support.
- [x] Checked every supplied Hugging Face entry for runtime, dependencies, license, languages, and cloning support.
- [x] Distinguished actual checkpoints from workflow/runtime repositories.
- [x] Ranked candidates by implementation difficulty.
- [x] Reduced the implementation scope to maintained, reusable integration paths.
- [x] Verified that workflow-node properties currently support static `hidden_in_inspector` metadata, but not model-dependent visibility.
- [x] Rechecked Chatterbox, Qwen3-TTS, F5-TTS, and the maintained `coqui-tts` XTTS v2 path against NodeTool's current dependency floor.
- [x] Ran resolver checks for the four packages with NodeTool's Torch, Transformers, Accelerate, Diffusers, safetensors, tokenizers, and NumPy constraints.

## Existing support

### `nodetool-huggingface`

- [x] `Bark`: `suno/bark` and `suno/bark-small` through the Transformers TTS pipeline.
- [x] `KokoroTTS`: `hexgrad/Kokoro-82M`, preset voices, multilingual G2P modes, speed, cancellation, and 24 kHz audio.
- [x] `TextToSpeech`: generic Transformers TTS with recommended Meta MMS checkpoints.
- [x] `HiggsAudio`: Higgs Audio v2 (`bosonai/higgs-audio-v2-generation-3B-base`), not Higgs TTS 3.
- [x] MiniMax-H3 joint video/audio generation exists, but not as a dedicated audio-only TTS node.
- [x] Fix and regression-test the Hugging Face local provider's non-Kokoro output normalization/yield path.
- [x] Advertise completed Bark, MMS, Supertonic, and F5 models from `get_available_tts_models()`.

### `nodetool`

- [x] The unified `nodetool.audio.TextToSpeech` node supports model, text, speed, and preset voice selection.
- [x] Transformers.js supports Kokoro and generic browser-side TTS/SpeechT5.
- [x] Cloud/provider packages already cover OpenAI, ElevenLabs, Gemini, MiniMax, Together, Replicate, fal, KIE, and other manifest-driven TTS endpoints.
- [x] Extend the unified TTS request for reference audio, reference transcript, language, and instructions.

## Dependency strategy

Updating Torch alone is not the main unlock. `nodetool-huggingface` already targets `torch==2.9.0`, `transformers>=5.15,<6`, `accelerate>=1.13,<2`, `diffusers>=0.39,<1`, `safetensors>=0.7,<1`, `tokenizers>=0.22.2,<1`, and NodeTool Core requires NumPy 2.x. Most blocked TTS packages pin older or simply different versions.

- [x] Confirmed that F5-TTS 1.1.22 resolves on Python 3.11 with NodeTool's current dependency constraints.
- [x] Confirmed that `coqui-tts` 0.27.5 resolves on Python 3.11 with those constraints, but has a reported package import failure on Transformers 5.1+ that the resolver cannot detect.
- [x] Confirmed that Chatterbox 0.1.7 cannot resolve: on Python below 3.14 it pins Torch/TorchAudio 2.6, and it also pins Transformers 5.2, Diffusers 0.29, safetensors 0.5.3, and NumPy below 2 on Python below 3.13.
- [x] Confirmed that Qwen3-TTS 0.1.1 cannot resolve because it pins Transformers 4.57.3 and Accelerate 1.12.0.
- [x] Found that the current lock can select TorchAudio 2.11 beside Torch 2.9; the official compatibility matrix requires TorchAudio 2.9.x with Torch 2.9.x.
- [x] Found that the committed `uv.lock` is stale relative to `pyproject.toml`: it also locks Transformers 5.5.4, Accelerate 1.10.1, Diffusers 0.35.1, safetensors 0.6.2, and tokenizers 0.22.1 below the currently declared minimums.
- [x] Regenerate and review `uv.lock` after aligning the Torch family, before running any model compatibility gate.
- [x] Pin Torch 2.9.0, TorchVision 0.24.0, TorchAudio 2.9.0, and F5's optional TorchCodec 0.9.1 to compatible releases; cross-platform wheel smoke tests remain release-CI work.
- [x] Prefer compatible optional extras over changing the shared ML stack for one model.
- [x] Do not downgrade Transformers, Diffusers, NumPy, or safetensors globally to admit one TTS package.
- [ ] When considering a global dependency upgrade, run the full Hugging Face image, video, audio, ASR, and 3D smoke matrix; a TTS-only pass is insufficient.

## Included models: easiest to hardest

This is the implementation order. A compatibility or license gate that fails moves that model out of scope rather than creating a custom environment.

| Order | Model/family | Cloning | Why it remains in scope | Stop condition |
|---:|---|---|---|---|
| 0 | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | No; preset voices | Already implemented; use as the provider and regression baseline. | None. |
| 1 | [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3) | No local zero-shot clone; preset styles only | Small CPU-capable ONNX model with maintained `supertonic` PyPI package and 31 languages. | Drop if the package conflicts with NodeTool's base environment. Do not integrate the external paid Voice Builder. |
| 2, stopped | [TeraSpace/TeraTTSv2](https://huggingface.co/TeraSpace/TeraTTSv2) | No; ten bundled styles | Self-contained ONNX graphs through a reviewable `AutoModel(..., trust_remote_code=True)` interface; English/Russian and streaming output. | Stopped: the model API/card still exposes no license metadata and the repository contains no license file as of the research date. |
| 3 | [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS) | Yes; reference audio + transcript | Maintained `f5-tts` PyPI package, direct inference API, a successful resolver check against NodeTool's current stack, chunk inference, and one reusable adapter. | Require a real import/synthesis test under Transformers 5.15 and the corrected Torch family. Surface the checkpoint's CC-BY-NC-4.0 license. |
| 4 | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | Yes; reference audio + transcript. Also voice design. | Maintained `omnivoice` package, compact 0.6B model, simple reusable API, and Transformers 5-compatible dependency range. | Drop if installation changes NodeTool's pinned core dependencies. Surface the model's CC-BY-NC license. |
| 5 | [openbmb/VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B) and [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | Yes; reference audio, optional transcript, controllable/high-fidelity modes in VoxCPM2 | One maintained `voxcpm` package and one shared adapter cover both checkpoints. VoxCPM2 also provides voice design and native streaming. | Drop both if the shared package cannot coexist with NodeTool dependencies. Do not fork the upstream implementation. |
| 6, conditional | [Audio8/Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b) | Yes; reference audio + exact transcript | Self-contained checkpoint, codec, processor, and Hugging Face remote code; 0.6B, 11 languages, 44.1 kHz. | Its card requests Transformers `<5`, while NodeTool targets Transformers 5. Test first. Drop it if it needs a compatibility fork, dependency downgrade, or isolated worker. |

## Deferred models: closest to reconsider first

These remain outside the implementation sequence, but are no longer buried in the general exclusion list. The order reflects estimated work to reach a supportable integration, not audio-quality preference.

| Deferred rank | Model/family | What makes it valuable | Current blocker and re-entry condition |
|---:|---|---|---|
| 1 | [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) through maintained `coqui-tts` | Mature API; reference-audio cloning without a transcript; 17 languages; preset and cached voices; 24 kHz; streaming. | The package resolves with NodeTool, supports Torch 2.9, and released an XTTS Transformers 5 fix, but an open package-wide import bug remains on Transformers 5.1+. Re-enter after a real import and synthesis smoke test passes on NodeTool's exact stack. The CPML checkpoint is non-commercial. |
| 2 | [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) | Strong tested quality; simple reference-audio cloning without a transcript; Multilingual V3 has 23+ languages; CFG/pace and exaggeration controls; MIT model/code license. | Version 0.1.7 hard-pins six shared dependency families to incompatible versions and includes a Git dependency. Re-enter when upstream relaxes those pins, or only if a bounded no-dependency compatibility spike passes without patching or vendoring upstream. |
| 3 | [Qwen3-TTS family](https://huggingface.co/collections/Qwen/qwen3-tts) | One official package covers preset voices, instruction control, voice design, streaming, and 0.6B/1.7B voice cloning across ten languages. | `qwen-tts` 0.1.1 pins Transformers 4.57.3 and Accelerate 1.12.0 and recommends an isolated environment. Re-enter after an upstream compatible release or a no-patch test on NodeTool's exact stack. Cloning requires a 0.6B/1.7B **Base** checkpoint; CustomVoice and VoiceDesign do not clone directly. |

## Explicitly out of scope

The following entries are intentionally ignored for this plan. They can be reconsidered only when upstream provides a compatible packaged/standard interface.

- [ ] [IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5): source-only, tightly pinned older environment, auxiliary models, custom license.
- [ ] [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro): large serving-oriented stack, incompatible pins, high VRAM, non-commercial license.
- [ ] [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512): recursive source checkout, incompatible pinned stack, system/proprietary optional dependencies.
- [ ] [netease-youdao/Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS): source-only Python 3.10/CUDA 12.6 environment plus external Amphion and auxiliary checkpoints.
- [ ] [Aratako/Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small): requires a Torch upgrade and Git dependencies.
- [ ] [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts): gated checkpoint and old, incompatible Parler-TTS dependency. It provides named voices/style prompting, not cloning.
- [ ] [bosonai/higgs-tts-3-4b](https://huggingface.co/bosonai/higgs-tts-3-4b): full cloning/streaming path currently centers on external SGLang/vLLM serving and has a non-commercial license.
- [ ] [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B): official standalone inference and cloning support are not currently reliable enough.
- [ ] [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m): Python path requires the large NeMo stack; cloning was deliberately removed.
- [ ] [javawock7618/comfy-MiniMax-H3-workflows](https://huggingface.co/javawock7618/comfy-MiniMax-H3-workflows): ComfyUI workflow JSON, not a model checkpoint. Do not translate it into a one-off NodeTool integration.
- [ ] [audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf): multi-model runtime repository, not one model. A new native `audio.cpp` backend is outside this reduced scope.

Unchecked boxes in this section mean “not implemented,” not pending work for this plan.

## Checkable implementation plan

### Phase 0 — shared TTS contract and baseline

#### `nodetool-core`

- [x] Create branch `feat/tts-reference-audio-contract` before editing.
- [x] Add optional `capabilities`, `languages`, `sample_rate`, and `requires_reference_text` fields to `TTSModel` with backward-compatible defaults.
- [x] Add optional `reference_audio`, `reference_text`, `language`, and `instructions` arguments to the provider TTS contract.
- [x] Update the Python worker handler to accept reference audio as bytes rather than a local path.
- [x] Add serialization and provider-handler tests.

#### `nodetool`

- [x] Create branch `feat/tts-cloning-inputs` before editing.
- [x] Extend TypeScript TTS model/request types with the new optional fields.
- [x] Update the TypeScript-to-Python bridge to send reference audio as a binary blob.
- [x] Add optional reference-audio, reference-text, language, and instruction pins to `nodetool.audio.TextToSpeech`.
- [x] Define one generic `json_schema_extra.visible_when` property rule that can inspect another property, including nested model fields and capability membership; do not hard-code model IDs in the UI.
- [x] Apply the same visibility evaluator in both the node body (`NodeInputs`) and Inspector property list.
- [x] Hide unconnected controls and handles that the selected model cannot use: `reference_audio` unless `voice_cloning`, `reference_text` unless `reference_transcript`, `language` unless `language_selection`, and `instructions` unless `voice_design` or `instruction_control`.
- [x] Let the existing `TTSModel` selector show preset voices only when the selected model advertises `preset_voice` and has a non-empty voice list, while retaining the legacy behavior for providers without capability metadata.
- [x] Preserve hidden values when switching models so switching back restores the user's setup; omit unsupported dormant values from provider requests.
- [x] If an input becomes unsupported while it has a connected edge, keep a compact disabled warning row and handle visible until the user disconnects it, rather than leaving an invisible edge.
- [x] Validate required conditional inputs after visibility is resolved (for example, require `reference_text` only when the selected cloning model sets `requires_reference_text`).
- [x] Add focused evaluator tests for capability changes and connected-edge fallback; both Inspector and node rendering consume the same tested evaluator.
- [x] Preserve existing workflows that only contain model, text, voice, and speed.
- [x] Add bridge, node, and backward-compatibility tests.

#### `nodetool-huggingface`

- [x] Fix and regression-test the non-Kokoro local-provider path so it yields valid audio.
- [x] Add shared helpers for reference-audio temporary files, resampling, output normalization, cancellation, and cleanup.
- [x] Add an end-to-end Kokoro test through the unified provider path.
- [x] Add one generic MMS/Bark provider test to prevent the existing silent-output regression.

### Phase 1 — preset/local CPU models

#### Supertonic 3

- [x] Verify `supertonic` 1.3.1 imports on NodeTool's exact pinned core stack without changing its versions.
- [x] Add a `supertonic` optional dependency extra.
- [x] Implement `SupertonicTTS` with model, text, language, preset style, and speed.
- [x] Preserve the model-reported sample rate.
- [x] Advertise 31 languages plus the `na` fallback and preset-voice capability.
- [x] Document that local reference-audio zero-shot cloning is not available (Voice Builder JSON is external).
- [x] Add mocked unit tests and an opt-in CPU synthesis smoke test.

#### TeraTTSv2

- [x] Check the current model card/API/repository for an explicit model license; none is present.
- [x] Mark TeraTTSv2 out of scope and stop at the failed license gate.
- [ ] Review and pin the exact Hugging Face remote-code revision if the license gate is resolved later.
- [ ] Verify the remote code works without vendoring or conflicting dependencies if reconsidered.
- [ ] Implement `TeraTTS` with language tags, preset voice, duration, sampler, and Russian stress controls.
- [ ] Support its streamed waveform output.
- [ ] Add unit tests and an opt-in CPU smoke test.

### Phase 2 — reusable voice-cloning adapters

#### F5-TTS

- [x] Add `f5-tts` as an optional dependency extra rather than a base dependency.
- [x] With the corrected matching Torch family, test the package import on Python 3.11 under NodeTool's exact Transformers 5.15 stack.
- [ ] Run one real synthesis on Python 3.11 and repeat the import/synthesis gate on Python 3.12 (the opt-in smoke requires a consented fixture and model download).
- [ ] If import or synthesis requires dependency downgrades, patches, vendoring, or an isolated environment, move F5-TTS to deferred and stop.
- [x] Implement `F5TTS` through its Python inference API, not its CLI or Gradio application.
- [x] Support reference audio plus a required reference transcript; do not silently load a separate ASR model when the transcript is empty.
- [x] Expose speed and the small set of stable inference controls; keep training, evaluation, and multi-speaker story tooling out of scope.
- [x] Preserve 24 kHz output and support upstream long-text chunking, cancellation, and temporary-file cleanup.
- [x] Advertise voice-cloning, bilingual base-model coverage, and CC-BY-NC-4.0 checkpoint-license metadata in the node/model presentation.
- [x] Add unit tests and an opt-in GPU cloning smoke test using a consented fixture.

#### OmniVoice

- [ ] Verify `omnivoice` installation against the locked environment.
- [ ] If it changes core dependency pins, mark OmniVoice out of scope and stop.
- [ ] Implement `OmniVoiceTTS` with text, language, reference audio, and required reference transcript.
- [ ] Add voice-design inputs supported by the same adapter.
- [ ] Return 24 kHz audio and support cancellation/cleanup.
- [ ] Advertise voice-cloning, voice-design, multilingual, and non-commercial capability/license metadata.
- [ ] Add unit tests and an opt-in GPU cloning smoke test using a consented fixture.

#### VoxCPM family

- [ ] Verify `voxcpm` installation against the locked environment.
- [ ] If it changes core dependency pins, mark both VoxCPM models out of scope and stop.
- [ ] Implement one shared adapter for VoxCPM 0.5B and VoxCPM2.
- [ ] Support plain TTS and reference-audio cloning for both checkpoints.
- [ ] Support optional transcript conditioning where accepted.
- [ ] Add VoxCPM2 voice design, controllable clone, high-fidelity clone, and streaming modes without duplicating model-loading code.
- [ ] Preserve 16 kHz output for VoxCPM 0.5B and 48 kHz output for VoxCPM2.
- [ ] Add unit tests and opt-in GPU smoke tests for both checkpoints.

### Phase 3 — conditional generic remote-code model

#### Audio8 Preview 0.6B

- [ ] Review and pin the executable Hugging Face remote-code revision.
- [ ] Run a bounded compatibility test with NodeTool's installed Transformers version.
- [ ] Verify plain synthesis and reference-audio cloning without dependency downgrades or patches.
- [ ] If compatibility fails, mark Audio8 out of scope and stop; do not fork or isolate it.
- [ ] Implement `Audio8TTS` using `AutoProcessor` and `AutoModel`.
- [ ] Require the exact reference transcript when cloning.
- [ ] Preserve 44.1 kHz output.
- [ ] Add unit tests and an opt-in GPU cloning smoke test.

## Deferred re-evaluation gates

### XTTS v2

- [ ] Re-test the latest `coqui-tts` release after the open Transformers 5.1+ package-import issue is fixed upstream.
- [ ] Verify `from TTS.api import TTS`, model download, one preset voice, one reference-audio clone, and a second run in the same worker on NodeTool's exact stack.
- [ ] Verify that the required TorchCodec build matches the selected Torch backend on Windows, Linux, and macOS.
- [ ] If all checks pass without patches or an isolated environment, move XTTS v2 into Phase 2 ahead of OmniVoice.
- [ ] If included later, expose language, preset speaker or reference audio, streaming, and cached cloned-voice IDs through one adapter and mark the CPML model as non-commercial.

### Chatterbox

- [ ] Watch for an upstream release that relaxes the exact Torch, TorchAudio, Transformers, Diffusers, safetensors, and NumPy pins and removes the unpinned Git dependency.
- [ ] When available, run a no-patch compatibility spike for English Chatterbox and Multilingual V3 under NodeTool's exact stack.
- [ ] Verify default synthesis, reference-audio cloning without transcript, all advertised languages, CFG/pace, exaggeration, cancellation, and repeated generation.
- [ ] If the spike passes, move Chatterbox into Phase 2 after F5-TTS and implement English/Multilingual variants through one package adapter.
- [ ] Do not downgrade NodeTool's shared stack or vendor a private Chatterbox fork to make it pass.

### Qwen3-TTS

- [ ] Watch for an upstream `qwen-tts` release compatible with NodeTool's Transformers and Accelerate ranges.
- [ ] Re-test package import and generation without FlashAttention first; keep FlashAttention an optional acceleration only.
- [ ] Treat CustomVoice, VoiceDesign, and Base as modes of one Qwen3-TTS family adapter rather than separate integrations.
- [ ] Verify preset voice, instruction control, voice design, streaming, and cloning with both the 0.6B and 1.7B Base checkpoints.
- [ ] Verify normal reference-audio-plus-transcript cloning and lower-quality `x_vector_only_mode` without a transcript.
- [ ] If the family passes without patches or an isolated environment, move it into Phase 2 after Chatterbox.
- [ ] Do not claim cloning for CustomVoice or VoiceDesign checkpoints; advertise cloning only for Base checkpoints.

## Common acceptance checklist

- [x] Every currently included model is returned by Hugging Face `get_available_tts_models()` with accurate capabilities.
- [x] Models without cloning never display or require cloning inputs.
- [x] All model-dependent fields use capability metadata rather than model-name checks.
- [x] Hidden model-dependent values survive model switching but are not sent to unsupported providers.
- [x] No connected input edge can become invisible after changing models.
- [x] Reference audio travels through the generic node and worker bridge without relying on a caller-local path.
- [x] Temporary reference files are removed after success, failure, and cancellation paths through a shared context manager.
- [x] Adapter output is validated as finite mono float32 and provider output is normalized to the existing 24 kHz PCM contract.
- [x] Empty, NaN, clipped, or malformed output produces an actionable error or safe clipping.
- [ ] A second run succeeds in the same worker without stale state or growing VRAM.
- [ ] A cancelled run does not break the next request.
- [x] Missing optional dependencies identify the exact install extra.
- [x] Custom, gated, or non-commercial licenses are visible in node/model documentation for the included adapters.
- [x] Generated package metadata is updated with `python -m nodetool.package_tools scan --write`.
- [x] Focused `ruff check` passes for all changed Python and test files.
- [ ] `black --check .` passes.
- [ ] `pytest -q` passes.
- [x] TypeScript build/typecheck and affected package tests pass in `nodetool` (a broader base-nodes build remains blocked by the checkout's missing `imapflow` dependency).

## PR sequence

- [ ] Open one `nodetool-core` PR containing the backward-compatible reference-audio/capability contract and its tests.
- [ ] Open one `nodetool` PR containing conditional input visibility, generic TTS cloning inputs, binary bridge transport, and tests.
- [ ] Open one `nodetool-huggingface` PR containing all model work that passes its gates.
- [ ] Keep provider repair, dependency alignment, shared helpers, and each model/family in separate commits inside the `nodetool-huggingface` PR.
- [ ] Do not open a separate PR per model or phase.

Keep every model/family independently reviewable at commit level while maintaining one PR per repository. A failed gate removes that model from the PR instead of expanding the architecture or dependency surface.
