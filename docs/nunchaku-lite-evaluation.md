# Evaluation: migrating from the Nunchaku engine to Nunchaku Lite

Status: evaluation, 2026-08-24. No behavior change in this document.

## TL;DR

Nunchaku Lite (diffusers ≥ 0.40.0) would eliminate our two biggest Nunchaku
pain points — custom per-torch/CUDA wheel distribution and the diffusers-drift
compat shims — and would delete most of `nunchaku_pipelines.py`. It is **not a
drop-in replacement today**: published Lite checkpoints only cover about half
of the model families we ship quantized nodes for, the checkpoints are
community-hosted, loading requires opting into remote kernel code execution
from an untrusted Hub publisher, and out-of-the-box per-step latency is worse
because the fused kernels are gone (recoverable mostly via `torch.compile`).

**Recommendation: migrate incrementally.** Add a Lite load path (nearly free —
plain `DiffusionPipeline.from_pretrained`) and Lite-based recommended models
for the covered families, keep the old engine for the uncovered ones, and only
remove the old path once checkpoint coverage exists (ideally mirrored under
`nodetool-ai` on the Hub).

## What we use the old engine for today

Loading code lives in `src/nodetool/huggingface/nunchaku_pipelines.py` (~585
lines) plus routing in `flux_utils.py`, `text_to_image_pipelines.py`,
`image_to_image_pipelines.py`, and per-node wiring. The pattern everywhere is:
download a single `svdq-{int4,fp4}_r32-*.safetensors` transformer file from a
`nunchaku-ai/*` repo, load it with a `Nunchaku*` model class, optionally load
the AWQ-quantized T5 encoder from `nunchaku-ai/nunchaku-t5`, then assemble the
rest of the pipeline from the base (often gated) repo.

Model families with quantized nodes:

| Family | Old checkpoint repo | Lite checkpoint available? |
|---|---|---|
| Flux.1 Schnell | `nunchaku-ai/nunchaku-flux.1-schnell` | ✅ `lite-infer/flux.1-schnell-nunchaku-lite-{int4,nvfp4}_r32-bnb4-text-encoder` |
| Flux.1 Dev | `nunchaku-ai/nunchaku-flux.1-dev` | ✅ `lite-infer/flux.1-dev-…` |
| Flux.1 Kontext Dev | `nunchaku-ai/nunchaku-flux.1-kontext-dev` | ✅ `lite-infer/flux.1-kontext-dev-…` |
| Flux.1 Fill Dev | `nunchaku-ai/nunchaku-flux.1-fill-dev` | ❌ none |
| Flux.1 Depth Dev | `nunchaku-ai/nunchaku-flux.1-depth-dev` | ❌ none |
| Flux.1 Canny Dev | `nunchaku-ai/nunchaku-flux.1-canny-dev` | ❌ none |
| Qwen-Image | `nunchaku-ai/nunchaku-qwen-image` | ✅ `lite-infer/Qwen-Image-…` (r32 and r128, int4 and nvfp4) |
| Qwen-Image-Edit (original) | `nunchaku-ai/nunchaku-qwen-image-edit` | ❌ none |
| Qwen-Image-Edit 2509 | `nunchaku-ai/nunchaku-qwen-image-edit-2509` | ✅ `lite-infer/qwen-image-edit-2509-…` (plus 4-step/8-step Lightning variants) |
| SDXL / SDXL-Turbo UNet | `nunchaku-ai/nunchaku-sdxl{,-turbo}` | ❌ none (Lite quantizes transformer linear layers via diffusers quantization config; no published SDXL UNet checkpoints) |
| T5-XXL text encoder | `nunchaku-ai/nunchaku-t5` (AWQ) | n/a — Lite repos bundle a bitsandbytes-4bit text encoder instead |

(Coverage snapshot from a Hub search for `nunchaku-lite`, 2026-08-24. Lite
also covers families we don't ship yet: Flux.2 Klein 4B/9B, Flux.1 Krea Dev,
Krea 2 Turbo, Z-Image Turbo, ERNIE-Image-Turbo, LTX-2.3 — potential new nodes
for free.)

## What a switch would remove

- **The wheel problem.** `requirements/nunchaku.txt` exists because the engine
  ships compiled wheels keyed to exact Python/torch/CUDA combinations, served
  from our own registry index, with a name collision against an unrelated PyPI
  package. Every torch bump risks breaking every user. Lite needs only
  `pip install kernels` (pure-Python package; the CUDA kernels are fetched and
  cached from the Hub at load time) plus diffusers ≥ 0.40.0 — our current pin
  is `>=0.39.0,<1`, so only the floor moves.
- **The compat shims.** `apply_qwen_rope_compat_shim()` and friends (~150
  lines plus a 195-line test file) exist because the engine subclasses
  diffusers internals and lags its signature changes (nunchaku-ai/nunchaku#884).
  Lite uses the *stock* diffusers model classes with generic quantized linear
  layers, so this whole class of breakage disappears.
- **Custom assembly.** `load_nunchaku_flux_pipeline` /
  `load_nunchaku_qwen_pipeline` (transformer + T5 + base-repo composition,
  `svdq`/precision filename validation, `set_attention_impl`,
  `set_offload`) collapse into a single
  `DiffusionPipeline.from_pretrained(model_id)` — the quantization config is
  read from the repo's `config.json` and handled by diffusers'
  `NunchakuLiteQuantizer`. The Lite path can share the ordinary non-quantized
  loading code, model-manager caching, and `apply_cpu_offload_if_needed`.
- **Gated-repo friction, partially.** Lite repos are full self-contained
  pipelines, so Schnell-family loads no longer touch the gated
  `black-forest-labs` base repos at all. (Dev-derived weights remain
  license-gated by BFL terms regardless of packaging — verify the lite-infer
  repos' license/gating state before recommending them.)

## What a switch would cost

- **Coverage gaps** (table above): Flux Fill/Depth/Canny, original
  Qwen-Image-Edit, and SDXL would lose their quantized variants unless we
  produce checkpoints ourselves with
  [diffuse-compressor](https://github.com/rootonchair/diffuse-compressor).
  SDXL's UNet likely stays unsupported (Lite targets linear layers in
  transformer-style models).
- **Per-step performance.** Lite drops the engine's model-specific fusions:
  QKV+RMSNorm+RoPE fusion is a 1.27× step-time factor and GELU-MLP fusion
  1.06× (Flux Schnell, RTX 5090, upstream numbers). `torch.compile` recovers
  much of it (compiled Lite NVFP4 is 1.8× over BF16 upstream), but compile
  adds per-shape warmup latency — a real UX cost in an interactive node
  editor where users change resolutions. Uncompiled Lite will be measurably
  slower per step than the current engine.
- **Remote-kernel trust.** Kernels load from
  `rootonchair/nunchaku-lite-kernels`, whose publisher is **not** a trusted
  kernel publisher on the Hub; diffusers therefore requires
  `DIFFUSERS_TRUST_REMOTE_KERNELS=true`, which downloads and executes code
  from the Hub on the user's machine. NodeTool setting that env var on users'
  behalf is a supply-chain decision that needs an explicit call (at minimum,
  pin a kernel revision if the `kernels` API allows it, and surface the
  opt-in in settings rather than defaulting it on).
- **Checkpoint provenance and durability.** `lite-infer` / `rootonchair` are
  community orgs (rootonchair authored the diffusers integration,
  huggingface/diffusers#14100), not the official `nunchaku-ai` org. If we
  depend on them, mirror the checkpoints under `nodetool-ai` on the Hub.
- **Re-downloads for existing users.** Old format = one ~7 GB transformer file
  reusing base-repo components already in the cache; Lite = a full pipeline
  repo including a bitsandbytes-4bit text encoder and VAE. Existing caches
  don't carry over, and `bitsandbytes` becomes a dependency (pip-installable,
  minor).
- **Low-VRAM path.** We currently use the engine's per-layer transformer
  offload (`set_offload(..., num_blocks_on_gpu=20)`) below ~18 GB VRAM,
  which runs in 3–4 GB. Lite models are standard diffusers modules, so the
  replacement is diffusers group/leaf-level offloading — needs validation
  that low-VRAM users don't regress.
- **Hardware support is a wash.** Lite kernels: sm_75/80/86/89/120, no Hopper
  (sm_90) — materially the same consumer-GPU matrix as the engine. NVFP4
  needs Blackwell + torch ≥ 2.7 + CUDA ≥ 12.8 on both.

## Recommended path

1. **Phase 1 — additive.** Bump diffusers floor to 0.40.0, add `kernels` (and
   `bitsandbytes` if absent) as optional deps, and add Lite-based recommended
   models for Flux Schnell/Dev/Kontext, Qwen-Image, and Qwen-Image-Edit-2509
   alongside the old entries. Detection is trivial: a repo whose
   `config.json` carries `"quant_method": "nunchaku_lite"` loads through the
   ordinary `from_pretrained` path — no `nunchaku_pipelines.py` involvement.
   Gate the `DIFFUSERS_TRUST_REMOTE_KERNELS` opt-in behind a NodeTool setting.
2. **Phase 2 — validate.** Benchmark Lite vs engine on representative GPUs
   (8 GB, 16 GB, 24 GB; int4 on Ada, nvfp4 on Blackwell), with and without
   `torch.compile`; validate the low-VRAM offload story and output quality
   (r32 vs r128 Qwen variants). Mirror chosen checkpoints under `nodetool-ai`.
3. **Phase 3 — converge.** If parity holds, mark old-engine model packs
   deprecated for covered families and either quantize the missing families
   with diffuse-compressor or keep the engine only for Fill/Depth/Canny,
   Qwen-Image-Edit-v1, and SDXL. Full removal of `nunchaku_pipelines.py`, the
   shims, `requirements/nunchaku.txt`, and the wheel registry entries only
   happens after that.

The engine remains actively maintained (v1.2.1, Jan 2026), so there is no
forcing event; the payoff is dependency simplicity and shim deletion, and the
risk is concentrated in checkpoint coverage/hosting and the remote-kernel
trust decision.
