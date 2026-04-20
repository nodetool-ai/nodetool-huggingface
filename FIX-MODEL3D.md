# Local 3D Generation - Review & Fix Plan

Scope: current implementation in `src/nodetool/nodes/huggingface/local_3d.py`
(target split: `text_to_3d.py` + `image_to_3d.py` + `_3d_common.py`) and the
related viewer in `nodetool/web/src/components/asset_viewer/Model3DViewer.tsx`.
Covers Shap-E, Hunyuan3D, StableFast3D, TripoSR, Trellis2, TripoSG. Does NOT
cover fal / kie / replicate.

---

## Platform support matrix

| Node | macOS install | macOS run | Linux+CUDA | Win+CUDA |
|---|---|---|---|---|
| `ShapETextTo3D` / `ShapEImageTo3D` | yes | **yes** (CPU/MPS, slow) | yes | yes |
| `Hunyuan3D` | **yes** (`hy3dgen.shapegen` is pure-Python; CUDA C++ ext lives in `hy3dgen/texgen/` which we never import) | no (pipeline assumes fp16 CUDA) | yes | yes |
| `StableFast3D` | **yes (with Xcode + libomp)** - `texture_baker` has explicit Metal branch, no `nvdiffrast` import | **experimental** (Metal-accelerated, untested) | yes | yes (build) |
| `TripoSR` | **yes (with Xcode)** - `torchmcubes` can fall back to CPU build | **experimental** (CPU mcubes, slow, untested) | yes | yes |
| `Trellis2` | no (`o_voxel` is Linux-only per upstream) | no | yes (24 GB min) | **no** |
| `TripoSG` | no (`diso` builds need CUDA, our code hard-gates CUDA) | no | yes | yes |

Only **Shap-E** is officially cross-platform at runtime. **SF3D and TripoSR**
are plausibly Apple-capable given upstream code paths, but neither is tested by
us or upstream.

---

## Status snapshot

- **Current code state:** implementation still lives in `local_3d.py`. The
  split to `text_to_3d.py` + `image_to_3d.py` + `_3d_common.py` should be next priority
- **Completed already:** the earlier quick fixes, doc / packaging cleanup,
  smoke-test guardrails, and supporting housekeeping are done.
- **Non-HF follow-up work:** generic `Model3D` node upgrades now live in
  `MODEL3D-CROSS-REPO.md`; future shared-type ideas live in
  `MODEL3D-CORE-FUTURE.md`.

---

## Execution order

Read this section top to bottom. The first unchecked group is the next work.

### 1. Next: fix model ownership and preload semantics

- [x] **Make `ModelManager` the single source of truth**
  - Do not keep long-lived strong refs like `self._model` / `self._pipeline`
    on node instances.
  - Preferred implementation: keep this scoped to `local_3d.py`.
  - Use `self._pipeline` only as a transient handle during `process()`, clear
    it in `finally`, and always re-fetch from `ModelManager`.
  - `preload_model()` should warm `ModelManager`, not pin models on `self`.
  - Keep `run_pipeline_in_thread()` / `move_to_device()` working by ensuring
    `self._pipeline` is set only for the active call path that needs it.
  - Do **not** refactor shared Hugging Face base code unless the local-only
    approach proves insufficient. If base-class changes are required, keep them
    minimal and limited to supporting transient pipeline handles.
  - Done means: no long-lived node-held model refs remain in the 3D nodes,
    preload still works, and the change does not broaden into unrelated HF
    infrastructure cleanup.
  - This should happen before the module split.

### 2. Then: split the module and regenerate metadata

- Goal: break up the oversized 3D module along clear modality lines without
  changing behavior, then refresh generated metadata to match the new layout.

- [x] **Split `local_3d.py` by modality**
  - `text_to_3d.py` for `ShapETextTo3D`
  - `image_to_3d.py` for the six image-to-3D nodes
  - `_3d_common.py` for shared helpers
  - Update smoke tests and imports.
  - Regenerate `package_metadata/nodetool-huggingface.json` after the split.

### 3. Then: finish packaging and install validation

- Goal: make the local 3D stacks installable in a predictable way without
  relying on unpublished dependency behavior.

- [x] **Add env-markered optional dependencies**
  Keep installable PyPI extras for pure-Python stacks and fail early only where
  platform constraints are real.

- **Note:** keep direct VCS deps out of published package metadata. Use pinned
  requirement files for git-only stacks instead.

- [x] **Validate install combinations**
  Confirm whether `[hunyuan3d,triposg]` resolves with `torch==2.9.0`; if not,
  document mutually-exclusive groups.

  **Results:** `[hunyuan3d]` and `[sf3d]` (PyPI companion deps only) resolve
  cleanly with `torch==2.9.0` + `transformers==5.5.4`.
  `[hunyuan3d,triposg]` fails to build `diso` without CUDA headers at install
  time — this is expected since `diso` is a CUDA C++ extension. On a CUDA system
  both extras are compatible. No mutually-exclusive groups exist.

- [x] **Pin git-only upstream commits**
  Maintain commit-pinned requirement files for `sf3d`, `triposr`, and
  `trellis2` so CI / local installs use known-good revisions.

### 4. Then: make exported `Model3DRef` assets consistent

- Goal: all local 3D generators should produce assets that behave the same way
  in the viewer and downstream nodes, without per-model special cases.

- [x] **Make GLB the canonical transport format**
  Make GLB the default internal/export transport for the local generators.
- [x] **Standardize orientation**
  Document `+Y` up once in `_3d_common.py`.
- [x] **Standardize centering / pivot**
  Default to bounding-box center at origin.
- [x] **Standardize metadata**
  Minimum metadata set: `seed`, `source_model`, `vertex_count`, `face_count`,
  `has_texture`, `units`, `orientation`.
- [x] **Extract a shared normalization / metadata helper**
  Route each local generator through one shared helper instead of ad hoc
  per-node export cleanup.
- [x] **Add one focused no-GPU contract test**
  Verify the shared export contract without loading real models.

### 5. Then: runtime warnings, low-VRAM rollout, and manual quality checks

- Goal: finish the remaining runtime-behavior work that affects how usable the
  shipped local 3D nodes are in practice.

- [x] **Roll out `low_vram_mode` on the remaining heavy nodes**
  Follow `D7`: `SF3D` and `TripoSG` try CPU offload when supported; `Trellis2`
  exposes the field but warns that upstream has no offload support.

- [ ] **Manual export / viewer verification**
  Confirm all supported local generators still preview correctly after the
  export-normalization changes.

### 6. Separate follow-up PR: Apple-experimental path

- Goal: keep the optional Apple-expansion work isolated from the core cleanup
  and packaging work.

- [x] **Soften CUDA gates in SF3D and TripoSR**
  Let the upstream libraries attempt non-CUDA execution instead of hard-failing
  early in our wrapper code, while keeping support explicitly experimental.

---

## Completed so far

Short reference only; these no longer need active planning detail.

- [x] **Core fixes:** dead-code cleanup, device fixes, export helper, rembg
  caching, seeding, preload rollout, smoke tests, and related runtime fixes
- [x] **Packaging / docs:** optional-deps cleanup, platforms docstrings,
  vendored upstream notes, revision pin scaffolding, disk-space preflight, and
  input-image validation
- [x] **Known packaging baseline:** PR #26 landed the first optional-dependency
  groups, but the remaining env-marker / pinning / install-validation follow-up
  stays in the execution order above
- [x] **Planning / guardrails:** smoke-test import discipline and other
  supporting cleanup decisions are in place
- [x] **Notable completed fixes:** node-held-ref cleanup where already landed,
  Hunyuan3D low-VRAM guardrails, TripoSG device / flash-decoder cleanup, and
  consistent `seed=-1` behavior are all captured in the current code / plan set

---

## Companion docs

This file now stays HF-only.

- `MODEL3D-CROSS-REPO.md` tracks concrete `nodetool` follow-up work for generic
  `Model3D` nodes, mesh cleanup, and library adoption.
- `MODEL3D-CORE-FUTURE.md` tracks future `nodetool-core` / `nodetool-sdk`
  ideas such as richer shared 3D asset types.

---

## Key decisions

Keep these stable unless the plan is intentionally re-scoped.

- **`D1`**: GLB is the default transport; OBJ stays opt-in.
- **`D2`**: split to `text_to_3d.py` + `image_to_3d.py` + `_3d_common.py`.
- **`D3`**: `seed=-1` means random; return the resolved seed in metadata.
- **`D4`**: `ModelManager` is the source of truth; node-held pipelines are
  transient only.
- **`D5`**: use normal published extras where possible, keep git-only stacks in
  pinned requirement files, and do not ship direct VCS dependencies in package
  metadata.
- **`D6`**: keep tests GPU-free in CI: smoke tests plus targeted mock-pipeline
  tests.
- **`D7`**: `low_vram_mode` is shared, but implementation is per-node.
- **`D8`**: Apple support for SF3D / TripoSR is experimental and should ship in
  a separate PR.
- **`D10`**: VRAM guidance is a warning, not a hard block.
- **`D12`**: `Model3DRef` outputs should share format, orientation, centering,
  and minimum metadata.
- **`D13`**: keep a small manual benchmark matrix for defaults.
- **`D14` / `D15`**: this plan ships the generation layer; generic cleanup work
  is a separate follow-up in `MODEL3D-CROSS-REPO.md`.

---

## Newer developments worth considering (April 2026 audit)

These are not part of the active execution order above.

### Models we should evaluate

- [ ] **Hunyuan3D-2.1 (Tencent, June 2025)**
  Recommended upgrade candidate. Shape + Paint pipeline, open-source PBR
  texture model, still under Tencent non-commercial terms.

- [ ] **Hunyuan3D-2.5 (Tencent, April 2025)**
  Investigate whether weights are actually available or API-only.

- [ ] **Direct3D-S2 (NeurIPS 2025, May 2025)**
  Track, but defer until a smaller / more practical upstream release exists.

- [ ] **ReLi3D (Stability AI)**
  Multi-view SF3D-style variant. Not critical unless user demand appears.

### Library version bumps

- [ ] **PyTorch 2.10**
  Evaluate in a dedicated PR; validate the broader HF stack first.
- [ ] **diffusers - investigate latest stable**
  0.36 had known issues; check for a patched release before bumping.
- [ ] **transformers 5.0**
  Defer until GA and wider ecosystem compatibility.
- [ ] **`hy3dgen` - re-pin for Hunyuan3D-2.1**
  Revisit if 2.1 adoption changes the required package line.

### What we are not missing

Audit-confirmed: Shap-E has had no major updates since 2023, TripoSR is
effectively superseded by TripoSG, and original Trellis is superseded by
Trellis2-4B.

---

## Out of scope

- Adding new local pipelines such as Zero123 / Stable Zero123 / InstantMesh /
  CRM / DreamGaussian.
- Texture-bake post-processing for Hunyuan3D shape-only output via
  `hy3dgen.texgen`.
- A full pluggable mesh post-processing graph with UV unwrap, retopology, and
  texture baking in this plan.
- OBJ/PLY loaders in `Model3DViewer.tsx` for now.

---

## General local HuggingFace execution improvements

Separate track, not 3D-specific.

- [ ] **`GHF1` - Split static metadata from live runtime availability**
- [ ] **`GHF2` - Standardize progress stages for long local inference jobs**
- [ ] **`GHF3` - Define cancellation and cleanup semantics**
- [ ] **`GHF4` - Improve local inference error taxonomy**
- [ ] **`GHF5` - Show warm vs cold start visibility**
- [ ] **`GHF6` - Revisit CPU-vs-CUDA execution pool strategy**
