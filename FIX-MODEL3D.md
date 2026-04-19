# Local 3D Generation – Review & Fix Plan

Scope: `src/nodetool/nodes/huggingface/text_to_3d.py` (and the related viewer in
`nodetool/web/src/components/asset_viewer/Model3DViewer.tsx`). Covers Shap-E,
Hunyuan3D, StableFast3D, TripoSR, Trellis2, TripoSG. Does NOT cover fal / kie /
replicate.

---

## Platform support matrix (verified against upstream sources)

| Node | macOS install | macOS run | Linux+CUDA | Win+CUDA |
|---|---|---|---|---|
| `ShapETextTo3D` / `ShapEImageTo3D` | yes | **yes** (CPU/MPS, slow) | yes | yes |
| `Hunyuan3D` | **yes** (`hy3dgen.shapegen` is pure-Python; CUDA C++ ext lives in `hy3dgen/texgen/` which we never import) | no (pipeline assumes fp16 CUDA) | yes | yes |
| `StableFast3D` | **yes (with Xcode + libomp)** — `texture_baker` has explicit Metal branch, no `nvdiffrast` import | **experimental** (Metal-accelerated, untested by us or upstream) | yes | yes (build) |
| `TripoSR` | **yes (with Xcode)** — `torchmcubes` `CMakeLists.txt` falls back to CPU build | **experimental** (CPU mcubes, slow, untested) | yes | yes |
| `Trellis2` | no (`o_voxel` is Linux-only per upstream) | no | yes (24 GB min) | **no** |
| `TripoSG` | no (`diso` builds need CUDA, our code hard-gates CUDA) | no | yes | yes |

Only **Shap-E** is officially cross-platform at runtime. **SF3D and TripoSR are
plausibly Apple-capable** given upstream code paths, but neither is tested on
Apple — by us or by upstream. The packaging strategy below reflects "ship +
let it try, don't pre-emptively block".

---

## Status

**PR #26 (`fix-model3d` branch, merged April 19 2026) landed against an earlier
revision of this plan.** It implemented `[x]` items in Quick-wins below
(#2, #3, #5, #9, #10, #13, #15, #16) plus a partial #8 and a single-file
rename to `local_3d.py`. The plan has since evolved:

- **D2 changed** from "rename to `local_3d.py`" → "split into
  `text_to_3d.py` + `image_to_3d.py` + `_3d_common.py`". PR-3 now contains
  corrective work (#1) to redo the rename properly.
- **#8 expanded** into nine sub-items (#8a–#8i) covering env markers,
  runtime installer, license/VRAM gating, etc. PR #26 only landed the
  basic optional-dependencies stanza — most of #8a–#8i are still open.
- All items not marked `[x]` are still pending.

---

## Quick-win priority (do these first)

- [x] **#2 – Delete dead `Trellis2._ensure_model_downloaded`**
  Lines `918–939`. Copy-paste from `Hunyuan3D`, references a non-existent
  `self.VARIANT_CONFIG`. Currently unreachable but will `AttributeError` if any
  refactor wires it up. Just delete it.

- [x] **#3 – Override `requires_gpu() -> False` on both Shap-E nodes**
  Shap-E's `process()` already supports CPU (lines 98, 227), but the base
  class default `requires_gpu() -> True` (huggingface_pipeline.py line 171)
  prevents the scheduler from running these on CPU/MPS-only machines.

- [x] **#10 – Cache `rembg` session in `ModelManager`** *(landed in PR #26)*
  `SF3D` (line 661) and `TripoSR` (line 800) call `rembg.new_session()` per
  invocation, re-loading U²-Net every time. Cache once with key
  `"rembg_u2net_session"`.

- [x] **#13 – Standardize seeding to per-call `torch.Generator`**
  Inconsistent today:
  - Shap-E: per-call `torch.Generator` (good).
  - Hunyuan3D / Trellis2: global `torch.manual_seed(...)` (leaks state across
    runs, lines 495, 988).
  - TripoSG: per-call `torch.Generator` (good).
  Pick per-call generators everywhere. See D3.

- [x] **#15 – Add `preload_model` to the 5 heavy nodes**
  Only `ShapETextTo3D` and `ShapEImageTo3D` implement `preload_model`.
  `Hunyuan3D`, `StableFast3D`, `TripoSR`, `Trellis2`, `TripoSG` all load
  lazily inside `process()`, so the executor's preload phase is a no-op for
  them and the first run pays full cold-start. **See C1 — must be reconciled
  with D4.**

- [x] **#9 – Resolve viewer/output format mismatch**
  `Model3DViewer.tsx` (lines 551–558) only renders GLB/GLTF, but Hunyuan3D /
  SF3D / TripoSR expose an OBJ output option. Either restrict outputs to GLB
  (simpler) or add `OBJLoader` / `PLYLoader` to the viewer (richer).
  See Decisions §D1.

- [x] **#5 – Extract a single `_export_mesh(mesh, format) -> bytes` helper**
  The "trimesh export, with optional `.cpu().numpy()` fallback" code is
  duplicated 5× with slight per-node variations (`include_normals=True` only
  for SF3D, `extension_webp=True` only for Trellis2). Centralize. **See C2 —
  the o_voxel-direct-to-GLB path stays separate.**

- [x] **#1 – Split module by input modality (HF `pipeline_tag` aligned)**
  Current state after PR #26: file lives at `local_3d.py`, namespace
  `huggingface.local_3d.*`. **PR went with the old D2 (single-file rename);
  new D2 is the modality split.** Pending follow-up in PR-3:
  - Split `local_3d.py` → `text_to_3d.py` (`ShapETextTo3D`) +
    `image_to_3d.py` (the other six nodes) + `_3d_common.py` (shared helpers).
  - Update imports in the existing smoke test and metadata regeneration path
    to match the split (mechanical).
  - No backward-compat alias layer is required; this is still effectively new
    functionality with no real external usage to preserve.
  See Decisions §D2.

- [x] **#16 – Add a no-GPU smoke test**
  At minimum: import the module, instantiate each node, assert
  `get_recommended_models()` returns valid `HuggingFaceModel`s. That single
  test would have caught finding #2.

---

## Dependency strategy (#8 — expanded)

Single biggest user-visible issue: today `pip install nodetool-huggingface`
declares `hy3dgen`, `pymeshlab`, `scikit-image` as **hard** dependencies, while
`sf3d`, `tsr`, `trellis2`, `o_voxel`, `rembg`, `diso` are not declared at all
and `ImportError` at runtime. We split this into A+C+D:

> **PR #26 status (April 2026):** Landed a basic optional-dependencies stanza
> with `sf3d`/`triposr`/`trellis2`/`triposg`/`all-3d` groups (no env markers,
> no version pins). This satisfies the *spirit* of #8 but does **not** address
> #8a (demote hard deps), #8b (env markers), #8c (runtime installer), #8d
> (commit pins), #8e (`is_available()` gating), #8f (Platforms docstrings),
> or #8i (CUDA gate softening). Most of the value of the dependency rework
> is still ahead of us.

- [x] **#8a – Demote heavy deps from hard to optional**
  Move `hy3dgen`, `pymeshlab`, `scikit-image` out of `[project] dependencies`.
  Goal: `pip install nodetool-huggingface` succeeds on a clean macOS box.

- [x] **#8b – Adopt env-markered `[project.optional-dependencies]` (Option A)**
  PR #26 added groups *without* env markers — replace with the version below
  so a clean macOS install of `[hunyuan3d]` succeeds and `[trellis2]` fails
  loudly only on Linux:
  ```toml
  [project.optional-dependencies]
  # Always installable — Shap-E only needs diffusers (already in core)
  shape = []

  # Pure-Python / wheel deps — install everywhere, runtime gates CUDA
  hunyuan3d = [
      "hy3dgen>=2.0.2,<2.1",
      "pymeshlab",
  ]
  triposg = [
      "diso ; platform_system != 'Darwin'",
      "rembg ; platform_system != 'Darwin'",
      "pymeshlab",
      "scikit-image",
  ]

  # Convenience meta
  all-3d-pypi = ["nodetool-huggingface[hunyuan3d,triposg]"]
  ```

- [ ] **#8c – Runtime installer for git-only stacks (Option C)**
  `sf3d`, `tsr`, `trellis2`+`o_voxel` are not on PyPI. Wire a `nodetool
  install-extra <name>` command (CLI + Electron action via
  `nodetool/electron/src/packageManager.ts`) that runs the right
  `uv pip install git+https://...@<commit>` with platform gating. For now,
  missing-install guidance should surface through clear node errors / logs,
  not new sidebar UI.
  Stacks: `sf3d`, `triposr`, `trellis2`.

- [ ] **#8d – Pin upstream commits for git-only deps (Option D)**
  Maintain `dev/requirements-3d-sf3d.txt` / `-triposr.txt` / `-trellis2.txt`
  with commit-pinned VCS URLs for CI / dev reproducibility. The runtime
  installer (#8c) reads these.

- [x] **#8e – Add static capability metadata per node**
  Add class-level metadata that can be safely emitted into package metadata
  and consumed later by shared product surfaces:
  - `SUPPORTED_PLATFORMS`
  - `INSTALL_HINT`
  - `license_warning`
  - `MIN_VRAM_GB`
  This keeps static facts like "Linux only" or
  "Install with `nodetool install-extra trellis2`" separate from live-machine
  runtime checks.

- [x] **#8f – `**Platforms:**` line in every heavy-node docstring**
  - Shap-E: *"Platforms: all (CPU/MPS/CUDA)."*
  - Hunyuan3D: *"Platforms: Linux+CUDA, Windows+CUDA. Installable on macOS but does not run."*
  - SF3D: *"Platforms: Linux+CUDA, Windows+CUDA. macOS Metal experimental (untested)."*
  - TripoSR: *"Platforms: Linux+CUDA, Windows+CUDA. macOS CPU experimental (slow, untested)."*
  - TripoSG: *"Platforms: Linux+CUDA, Windows+CUDA (build required)."*
  - Trellis2: *"Platforms: Linux+CUDA only, 24 GB+ VRAM."*

- [x] **#8i – Soften CUDA gates in SF3D and TripoSR**
  Today both raise `RuntimeError("requires CUDA")` (lines 638-640, and TripoSR
  similarly) before the upstream code even gets a chance. Replace with a
  permissive device check + warning when running on non-CUDA, and let the
  underlying library fail with its own (more informative) error. This unlocks
  the experimental Apple paths above.

- [x] **#8g – Skip Option B (direct VCS deps in `pyproject.toml`)**
  PyPI rejects wheels with VCS deps in metadata; would block our publish
  pipeline. Documented for posterity.

- [ ] **#8h – Validate combined install resolves**
  Confirm `[hunyuan3d,triposg]` resolves with our pinned `torch==2.9.0`. If
  not, document mutually-exclusive groups.

---

## Correctness fixes

- [x] **#4 – Shap-E ignores the device chosen by `_resolve_hf_device`**
  `load_model` may place weights on `mps` (preferred on Apple Silicon by
  `local_provider_utils._resolve_hf_device`), but Shap-E's `process()` re-does
  `"cuda" if torch.cuda.is_available() else "cpu"` at lines 98, 114, 227,
  249. On macOS the `torch.Generator` ends up on `cpu` while the model is on
  `mps`. Drive everything off `self._pipeline.device`.

- [x] **#6 – Stop holding strong refs on the node instance**
  *(Implemented for all heavy nodes: Hunyuan3D, SF3D, TripoSR, Trellis2, TripoSG.
  ShapE nodes still use self._pipeline via HuggingFacePipelineNode base class —
  requires upstream nodetool-core change.)*
  Pattern `self._model = ModelManager.get_model(cache_key) or new_model;
  ModelManager.set_model(...)` keeps a strong reference on `self._model`. If
  `ModelManager` evicts under VRAM pressure, the GPU memory still can't be
  freed because the node instance pins it. Always read through
  `ModelManager.get_model(cache_key)` and don't stash on `self`.
  **See G6 — reconcile with `move_to_device`.**

- [x] **#7 – Hunyuan3D `low_vram_mode` monkey-patch may break silently**
  Lines 476–487 fake a `.components` dict on the `hy3dgen` pipeline before
  calling `enable_model_cpu_offload()`. That helper relies on more than just
  `.components` (hooks, `_execution_device`, `_offload_gpu_id`). Wrap in a
  try/except with a clear "low_vram_mode unavailable for this hy3dgen
  version" warning, and pin the validated `hy3dgen` version range (G7).

- [x] **#11 – TripoSG `_prepare_image` hard-codes `.cuda()`**
  Lines 1181–1234 use literal `.cuda()`. Currently fine because `process()`
  gates on CUDA, but brittle. Use `device` consistently.

- [x] **#12 – TripoSG always passes `flash_octree_depth`**
  Even when `use_flash_decoder=False` (line 1349). Verify upstream behavior;
  at depth 9 this can blow VRAM if the pipeline preallocates flash buffers
  unconditionally.

- [x] **#14 – `seed=-1` semantics inconsistent**
  Most nodes treat `-1` as "no generator". TripoSG defaults to `42` and
  converts `-1` to a freshly random seed it never echoes back. Pick one
  convention; if random is allowed, surface the actually-used seed in
  output metadata for reproducibility. See Decisions §D3.

---

## Polish / UX

- [x] **#17 – Document vendored `triposg` upstream**
  `src/triposg/` is excluded from ruff but ships in the wheel. Add
  `src/triposg/UPSTREAM.md` with the upstream commit SHA and any local
  patches so future merges are tractable.

- [ ] **#18 – Add `low_vram_mode` to all heavy nodes**
  Currently only `Hunyuan3D` exposes it. Trellis2 (24 GB), SF3D (~6 GB), and
  TripoSG (~8 GB) could expose `enable_sequential_cpu_offload()` /
  `enable_model_cpu_offload()` behind the same flag. See D7.

- [x] Hunyuan3D `_ensure_model_downloaded` docstring still says "75 GB repo"
  (line 412). Make sure the message matches reality after the
  `allow_patterns` fix.

- [x] Verify all heavy nodes set
  `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")`
  the way Trellis2 does (line 948) — fragmentation hurts every long-lived
  CUDA worker, not just Trellis2.

---

## Gaps surfaced during planning

- [ ] **G1 – Regenerate `package_metadata/nodetool-huggingface.json`**
  After #1 rename and #8 dependency / `is_available` changes, the package
  metadata JSON must be regenerated. Run the regen script and verify only
  intended diffs.

- [x] **G2 – No backward-compat layer required for the module split**
  We can ignore namespace migration complexity here. There is no meaningful
  external usage to preserve yet, so `local_3d.py` can be split cleanly into
  `text_to_3d.py` and `image_to_3d.py` without alias machinery in
  `nodetool-core`.

- [x] **G3 – Do not bake runtime availability into generated metadata**
  **Verified: `nodetool-sdk/csharp/Nodetool.Types/scripts/generation/discovery.py`
  imports node classes during build-time discovery.** That means package
  metadata can safely contain static facts (`SUPPORTED_PLATFORMS`,
  `INSTALL_HINT`, `license_warning`, `MIN_VRAM_GB`) but should not pretend to
  know runtime facts like "package installed on this machine" or "current GPU
  has enough VRAM". Those belong to a live local-HF runtime layer, which is
  tracked separately at the end of this file.

- [ ] **G4 – Verify VCS-URL behavior in `pyproject.toml`**
  Confirms #8g: PyPI rejects sdists/wheels with `direct_url` deps in
  metadata. Hence Option C (runtime installer) for git-only stacks.

- [ ] **G5 – CUDA / torch version conflict audit**
  Each git-only pipeline has its own `requirements.txt` pinning torch
  differently. Run a clean install of each combo with our `torch==2.9.0`
  and document mutually-exclusive groups before #8c lands.
  *Note: this is empirical — can only be verified during PR-3 implementation,
  not via static review. Flag in the PR description so we don't get
  blindsided.*

- [ ] **G6 – Reconcile #6 (no `self._model`) with `move_to_device`**
  `HuggingFacePipelineNode.move_to_device()` (line 133) walks
  `self._pipeline.to(device)`. Either:
  - keep `self._pipeline` as a *transient* handle set at top of `process()`
    and cleared in `try/finally`, or
  - rework `move_to_device` to look up via `ModelManager` using the cache
    key.
  Pick before refactoring #6.

- [x] **G7 – Pin `hy3dgen` version**
  Today `hy3dgen>=2.0.2` (open upper bound). After validation, pin
  `hy3dgen>=2.0.2,<2.1`.

---

## Conflicts inside the plan

- [ ] **C1 – `#15` (preload_model) vs `D4` (no `self._model`)**
  If D4 forbids stashing on `self`, what does `preload_model` do?
  - Option A *(recommended)*: `preload_model` warms the `ModelManager` cache
    keyed by class; `process()` always re-fetches.
  - Option B: `preload_model` stashes on `self`, `process()` clears in
    `try/finally`.
  Spell this out so the 5 new `preload_model` implementations are
  consistent.

- [ ] **C2 – `#5` helper signature**
  Trellis2's primary path goes `o_voxel.postprocess.to_glb(...)` with kwargs
  the trimesh helper can't accept. Resolution: helper covers only the
  trimesh path; Trellis2 keeps its `o_voxel` call site, falls back to the
  helper on exception. SF3D's `include_normals=True` becomes a kwarg the
  helper accepts as `**export_kwargs`.

---

## Additional findings (round 3)

- [x] **#19 – Pin model revisions in `from_pretrained`**
  Added `MODEL_REVISIONS` table + `_model_revision()` helper. All
  `snapshot_download` / `from_pretrained` calls now pass `revision=`.
  Values are currently `None` (latest) — fill in verified SHAs and bump
  intentionally.

- [x] **#20 – SF3D licensing surfaced in node UI**
- [x] **#20 – SF3D licensing surfaced in node UI**
  Surfaced as `license_warning: ClassVar[str | None]` on SF3D, Hunyuan3D,
  and Trellis2. Emitted in node logs on first model load. MIT nodes have
  `None`.

- [x] **#21 – Disk-space pre-flight for model downloads**
  Added `_check_disk_space(estimated_gb, cache_dir=)` helper. All
  heavy-node load paths call it before `snapshot_download`. Raises `OSError`
  with a human-readable message.

- [ ] **#22 – Surface VRAM budgets cleanly**
  Break this into explicit, checkable deliverables:
  - [x] **#22a** Every heavy node exposes `MIN_VRAM_GB: ClassVar[int]`
    and `ESTIMATED_DOWNLOAD_GB: ClassVar[float]`.
  - [ ] **#22b** At runtime, compare available VRAM against `MIN_VRAM_GB`
    using the existing/shared hardware probe where available (or a thin shared
    adapter), and emit a soft warning in node logs when the machine is under
    the recommended budget.
  - [ ] **#22c** Add at least one focused test for the warning path and one
    manual verification note confirming the warning appears without blocking
    execution.

- [x] **#24 – Input image validation**
  Each `process()` calls `await context.asset_to_io(self.image)` then opens
  with PIL. Failure modes (corrupt image, EXIF-rotated, palette-only) all
  bubble as raw `PIL.UnidentifiedImageError`. Wrap once in a helper that
  returns a friendly `ValueError("Invalid input image: <reason>")`.

- [ ] **#25 – Make exported `Model3DRef` assets consistent across generators**
  Break this into explicit, checkable deliverables:
  - [ ] **#25a** Choose and document the canonical transport format (`GLB`) for
    generated assets inside nodetool.
  - [ ] **#25b** Standardize exported assets on a single orientation convention:
    `+Y` up, forward axis documented once in `_3d_common.py`, and apply that
    convention in each local generator where upstream export hooks allow it.
  - [ ] **#25c** Standardize exported assets on a single centering / pivot
    convention: bounding-box center at origin by default, with optional
    scale-normalization-to-unit-box available but not forced.
  - [ ] **#25d** Define the minimum shared metadata set on `Model3DRef`:
    `seed`, `source_model`, `vertex_count`, `face_count`, `has_texture`,
    `units`, `orientation`.
  - [ ] **#25e** Extract a shared normalization / metadata helper into
    `_3d_common.py` and route each local generator through it.
  - [ ] **#25f** Add at least one no-GPU test that asserts the shared metadata
    contract on a stubbed generated model.
  - [ ] **#25g** Add one manual verification checklist item that confirms all
    supported local generators preview correctly in the viewer after export and
    appear upright / centered without per-model viewer hacks.

- [ ] **#26 – Add a small manual evaluation matrix for defaults**
  Smoke tests catch wiring bugs, not product quality regressions. Maintain a
  lightweight benchmark sheet for 3-5 canonical prompts/images recording:
  runtime, peak VRAM, output size, viewer success, mesh quality, and texture
  quality. Use it to justify default models, recommended settings, and future
  upgrades like Hunyuan3D-2.1.

- [ ] **#27 – Make the phase-1 product boundary explicit**
  This plan delivers a solid **local 3D generation layer** inside nodetool:
  one input, one generated 3D asset, consistent preview/export behavior, clear
  install/runtime guidance. Mesh cleanup, retopology, UV unwrap, texture-bake,
  and multi-output 3D workflows remain a deliberate follow-up track rather
  than implicit scope creep inside these PRs.

---

## Notes for downstream / cross-repo concerns

- [ ] **N1 – Big GLBs and the WebSocket runner**
  `nodetool/packages/websocket/src/unified-websocket-runner.ts` serializes
  outputs back to the UI. Trellis2 with `texture_size=4096` +
  `decimation_target=1M` easily produces 40+ MB GLBs. Confirm those route
  through asset storage (out-of-band) and not inline WS frames before
  shipping #18 (which encourages cranking texture sizes).

- [ ] **N2 – CI matrix for `#16`**
  Smoke tests must not import `hy3dgen` / `sf3d` / `tsr` / `trellis2` /
  `triposg` at module top-level — CI doesn't have them. Current code already
  defers those imports inside `process()`; keep it that way after the
  refactor.

- [ ] **N3 – Asset-store filename collisions**
  All `model3d_from_bytes` calls use `f"{prefix}_{self.id}.{format}"`. If a
  node runs twice in the same workflow (loop), the second result may
  overwrite the first depending on asset-store semantics. Verify
  `processing_context.model3d_from_bytes` either deduplicates names or
  generates unique IDs internally.

---

## Decisions (answers to open design questions)

### D1 – Viewer vs OBJ output (finding #9)

**Decision: restrict outputs to GLB by default; keep OBJ as an opt-in.**

- Default `output_format` to `GLB` on Hunyuan3D, SF3D, TripoSR.
- Keep the OBJ option for users who need it for downstream tooling, but add a
  hint in the field description: *"GLB is recommended; OBJ files are not
  previewable in the canvas viewer."*
- Do NOT invest in OBJ/PLY loaders in `Model3DViewer.tsx` right now — GLB is
  the canonical exchange format for the rest of the stack and adds proper
  texture support.

### D2 – Module / namespace split (finding #1)

**Decision: split into `text_to_3d.py` + `image_to_3d.py`, mirroring
HuggingFace's official `pipeline_tag` taxonomy.**

Rationale:

- HF Hub uses canonical task names `text-to-3d` and `image-to-3d` as
  `pipeline_tag` values (Computer Vision section). Aligning our module names
  with these tags makes node discovery, model filtering, and future model
  additions (text→3D in particular) intuitive for anyone familiar with HF.
- We also already use this `<modality>_to_<modality>` convention elsewhere
  (`text_to_image.py`, `image_to_image.py`, `text_to_video.py`), so this
  keeps the `nodes/huggingface/` directory internally consistent.
- The single-file alternative (`local_3d.py`) was considered but rejected:
  it hides the input-type distinction in the namespace and offers no clear
  upside as the file already exceeds 1300 lines.
- The `generate_3d.py` alternative was rejected for the same reasons —
  it does not communicate input modality and breaks the established naming
  convention.

**Resulting layout:**

```text
src/nodetool/nodes/huggingface/
  text_to_3d.py       # ShapETextTo3D (+ future text→3D nodes)
  image_to_3d.py      # ShapEImageTo3D, Hunyuan3D, StableFast3D,
                      # TripoSR, Trellis2, TripoSG (+ future image→3D)
  _3d_common.py       # shared helpers: device resolution, mesh export,
                      # cached rembg session, Generator factory
```

**Resulting node namespaces:**

- `huggingface.text_to_3d.ShapETextTo3D`
- `huggingface.image_to_3d.ShapEImageTo3D`
- `huggingface.image_to_3d.Hunyuan3D`
- `huggingface.image_to_3d.StableFast3D`
- `huggingface.image_to_3d.TripoSR`
- `huggingface.image_to_3d.Trellis2`
- `huggingface.image_to_3d.TripoSG`

**Migration / scope note:**

- No backward-compat aliasing is required for this split.
- Shared helpers extracted to `_3d_common.py` to avoid the duplicated
  imports / device resolution / rembg cache / mesh export code that the
  single-file alternative was trying to avoid.

### D3 – Seeding convention (finding #14)

**Decision:**

- `-1` always means *random*.
- All nodes use a per-call `torch.Generator(device=...)`.
- When the seed is random, generate it with `torch.randint(...)` once,
  pass it to the generator, and **return the resolved seed in the
  output `Model3DRef.metadata.seed`** so the user can re-run deterministically.
- Default seed value across all nodes: `-1` (drop TripoSG's `42` default for
  consistency).

### D4 – Where to cache shared assets (findings #6, #10)

**Decision: `ModelManager` is the single source of truth.**

- Nodes hold a *transient* convenience handle (`self._pipeline = ...` only
  for the duration of one `process()` call, cleared in `finally`) and always
  re-fetch via `ModelManager.get_model(key)` at the top of `process()`.
- Shared subsystems get their own keys: `"rembg_u2net_session"`,
  `"briaai/RMBG-1.4_BriaRMBG"`, etc.
- **Conflict C1** dictates that `preload_model` warms the `ModelManager`
  cache and does NOT assign to `self._pipeline`.

### D5 – Optional dependencies packaging (finding #8)

**Decision: A + C + D combo (see expanded #8a–#8h above).**

- **A** for everything on PyPI (env markers + `[project.optional-dependencies]`).
- **C** for git-only stacks (sf3d, triposr, trellis2): runtime installer
  command surfaced in the UI via `packageManager.ts`.
- **D** for our own dev / CI reproducibility (commit-pinned requirements
  files).
- **Skip B** for published wheels (PyPI rejects VCS metadata).
- Static metadata exposes `SUPPORTED_PLATFORMS`, `INSTALL_HINT`,
  `license_warning`, and `MIN_VRAM_GB`. Live runtime checks for install state,
  device state, cancellation, and progress are tracked separately in the
  cross-cutting local-HF section at the end of this file.
- Hunyuan3D moves *out* of the git-only stack: `hy3dgen` is a normal PyPI
  package, the CUDA C++ extensions only ship under `hy3dgen/texgen/` which
  we never import.

### D6 – Test strategy (finding #16)

**Decision: two layers, neither requires a GPU.**

1. *Module smoke test* — `pytest tests/test_local_3d_smoke.py`: import the
   module, instantiate every node, validate `get_recommended_models()`,
   `get_basic_fields()`, `get_title()`, and that all field defaults pass
   pydantic validation. Catches dead-code bugs like #2.
2. *Mock-pipeline test* — monkey-patch `ModelManager.get_model` to return a
   stub pipeline that yields a fixed trimesh, and assert `process()` returns
   a `Model3DRef` with the expected format. Catches export/wiring bugs
   without ever loading a real model.

GPU integration tests stay out of CI; document them under `notebooks/` as
"manual smoke runs". Tests must defer all heavy imports inside functions
(N2).

### D8 – Apple Silicon support stance (findings #8i, SF3D/TripoSR audit)

**Decision: "experimental, opt-in, no support promises".**

- Soften CUDA gates in SF3D and TripoSR (#8i).
- Docstrings and platform table say "experimental, untested" for those two
  on macOS.
- Do **not** add Apple Silicon to CI — cost is high, payoff low.
- If a user reports it works, promote to "supported"; if it breaks, the error
  surfaces from upstream (and we can point to upstream issues).
- Hunyuan3D, Trellis2, TripoSG remain CUDA-only.

### D9 – Model-revision pinning (finding #19)

**Decision: pin SHAs in a central `MODEL_REVISIONS` table, refresh quarterly.**

```python
# nodetool/nodes/huggingface/local_3d.py
MODEL_REVISIONS: dict[str, str] = {
    "openai/shap-e": "abc123...",
    "tencent/Hunyuan3D-2": "def456...",
    ...
}
```

- All `from_pretrained` / `snapshot_download` reads from this table.
- A bare CI job runs once a week to diff against current HEAD revisions and
  open an issue if anything moved (so we notice upstream churn deliberately).
- Acceptable risk: pinning means users don't get upstream fixes
  automatically — explicit bumps as part of `nodetool-huggingface` releases.

### D10 – VRAM budget surfacing (finding #22)

**Decision: class attribute + runtime log warning, no hard block. This
decision is only "done" when #22a-#22c are all checked off.**

- Each heavy node declares `MIN_VRAM_GB: ClassVar[int]`.
- Static metadata advertises the recommended VRAM budget.
- Reuse nodetool's existing runtime GPU detection path for CUDA / live VRAM
  information where available; do not add a second independent hardware-probe
  system just for these 3D nodes.
- If runtime device information is available, emit a soft warning in node logs
  when the machine is below the recommended VRAM budget.
- We don't hard-block because users with `low_vram_mode=True` may make it
  work, and we shouldn't over-promise either way.

### D11 – Licensing surfacing (finding #20)

**Decision: `license_warning: ClassVar[str | None]` on each node, surfaced in
logs / docs for now.**

- Default `None` (no warning) — e.g. Shap-E MIT, TripoSR MIT, TripoSG MIT.
- Set on **Hunyuan3D** (Tencent non-commercial), **SF3D** (Stability
  Community License, $1M revenue cap), **Trellis2** (Microsoft Research
  License, non-commercial) with concise human-readable text + upstream
  license URL.
- **No "I accept" gate** and no new sidebar UI in this plan. Revisit richer
  product surfaces later if the warnings prove useful enough.

### D12 – Canonical `Model3DRef` contract (finding #25)

**Decision: exported assets should look consistent across generators, and this
decision is only "done" when #25a-#25g are all checked off.**

- GLB is the canonical transport format inside nodetool.
- All exporters should converge on the same orientation convention: `+Y` up,
  with the forward-axis convention documented once in `_3d_common.py`.
- All exporters should converge on the same default centering convention:
  bounding-box center at origin.
- Preserve per-model output controls, but normalize the final asset enough that
  the viewer and downstream nodes do not need model-specific special cases.
- Include a minimum shared metadata set on `Model3DRef`: `seed`,
  `source_model`, `vertex_count`, `face_count`, `has_texture`, `units`,
  `orientation`.
- Scale normalization remains opt-in rather than forced, so advanced users can
  preserve native dimensions when needed.

### D13 – Evaluation strategy for defaults (finding #26)

**Decision: keep CI light, but require a small manual benchmark matrix.**

- CI covers smoke/import/export wiring only.
- Product choices (recommended models, default output settings, whether a
  model is worth shipping) are validated on 3-5 canonical prompts/images.
- Record runtime, peak VRAM, output size, viewer success, mesh quality, and
  texture quality in the plan/PR notes whenever defaults change.
- New model additions should include at least one documented successful local
  run on a supported machine before becoming recommended.

### D14 – Product boundary for this work (finding #27)

**Decision: this plan ships the generation layer, not the full 3D toolchain.**

- In-scope: local text/image → 3D generation, sane exports, viewer-safe
  outputs, install/runtime guidance, and basic metadata/warnings.
- Out-of-scope for these PRs: retopology, UV unwrap, cleanup pipelines,
  texture baking beyond what the upstream generator already emits, and
  multi-asset 3D workflows.
- Those are follow-up product tracks that should consume `Model3DRef` rather
  than complicate the generation nodes themselves.

### D15 – Build on the existing generic `nodetool.model3d.*` surface

**Decision: post-processing follow-up should upgrade existing generic nodes,
not invent a parallel 3D toolchain from scratch.**

- `nodetool/packages/base-nodes/src/nodes/model3d.ts` already exposes generic
  nodes such as `Decimate`, `Transform3D`, `CenterMesh`,
  `RecalculateNormals`, `FlipNormals`, `MergeMeshes`, `Boolean3D`,
  `FormatConverter`, and `GetModel3DMetadata`.
- The right follow-up is to audit which of those are already product-ready,
  upgrade the weak ones, and add a small number of missing high-value cleanup
  nodes for AI-generated meshes.
- Priority order: real decimation / metadata / conversion, normalization,
  largest-component cleanup, then mesh repair.
- Full UV unwrap, texture baking, and retopology stay later-phase because
  they are much heavier and less reliable than the core cleanup steps above.

### D7 – `low_vram_mode` rollout (finding #18)

**Decision: ship behind a single shared `low_vram_mode` field on each heavy
node, but implement per-node:**

- `Hunyuan3D`: keep current monkey-patched `enable_model_cpu_offload` (with
  try/except hardening from #7).
- `SF3D`, `TripoSG`: call `enable_model_cpu_offload()` if the pipeline
  supports it, else no-op with a log warning.
- `Trellis2`: **confirmed no upstream offload support** — `Trellis2ImageTo3DPipeline`
  inherits from a custom `Pipeline` base, not `diffusers.DiffusionPipeline`,
  so `enable_*_cpu_offload` doesn't exist. Field present but no-op + warning;
  docstring directs users to Hunyuan3D Mini for low-VRAM use cases.

---

## Suggested PR slicing

- [x] **PR-1 "Quick fixes"** — *partially landed via PR #26 + scope drift*
  - [x] #2 delete dead code — *PR #26*
  - [x] #3 Shap-E `requires_gpu` — *PR #26*
  - [ ] #4 Shap-E device fix — **still pending, slipped past PR #26**
  - [ ] #11 TripoSG `_prepare_image` cleanup
  - [ ] #19 + D9 pin model revisions (table only, no behavior change)
  - [ ] #24 input-image validation helper
  - [ ] Hunyuan3D docstring cleanup
  - [x] #16 + N2 smoke test — *PR #26 (file: `tests/test_local_3d_smoke.py`)*
  - [ ] #17 vendored `triposg/UPSTREAM.md`

  **PR #26 also incidentally landed PR-2 work** (#5, #10, #13, #15) and a
  partial #1/#8 (single-file rename + basic optional deps). Treat PR-1.5
  below as a small follow-up to mop up the items the PR missed.

- [x] **PR-1.5 "PR #26 follow-ups"** *(small, low risk)*
  - [x] #4 Shap-E device fix (slipped from PR-1) — use `self._pipeline.device`
    and `_resolve_device()` helper; MPS generator uses "cpu" device
  - [x] #11 TripoSG `_prepare_image` cleanup — accept `device` kwarg,
    replace all `.cuda()` with `.to(device)`
  - [x] Hunyuan3D docstring cleanup — updated `_ensure_model_downloaded`
    docstring to say "~5 GB standard, ~2 GB mini" instead of "75 GB"
  - [x] #14 verify per-call seed metadata is actually returned in
    `Model3DRef.metadata` (D3) — all seeded nodes now pass
    `metadata={"seed": seed, "source_model": ...}` to `model3d_from_bytes`
  - [x] #19 + D9 pin model revisions — `MODEL_REVISIONS` table +
    `_model_revision()` helper; values are `None` (latest), fill in SHAs
  - [x] #24 input-image validation helper — `_open_pil_image()` wraps
    PIL open+load with friendly `ValueError` for corrupt/unsupported files
  - [x] #17 vendored `triposg/UPSTREAM.md`
  - [x] Verify `_export_mesh` helper from PR #26 covers Trellis2's
    `extension_webp=True` and SF3D's `include_normals=True` paths (C2)
    — confirmed: SF3D passes `include_normals=True`, Trellis2 uses its own
    `o_voxel.postprocess.to_glb()` path with `_export_mesh` as fallback

- [ ] **PR-2 "Refactor & helpers"** *(reduced scope after PR #26)*
  - [x] #5 + C2 export helper — *PR #26 (verify C2 handled — see PR-1.5)*
  - [x] #10 rembg cache — *PR #26*
  - [ ] #6 + G6 ModelManager refactor — **the more nuanced change; PR #26
    only added `preload_model`, did not stop pinning `_pipeline` on `self`.**
    Still need to read through `ModelManager.get_model(cache_key)` instead
    of `self._pipeline`.
  - [x] #13 + D3 seeding rework — *PR #26 (verify #14 metadata in PR-1.5)*
  - [x] #15 + C1 `preload_model` rollout — *PR #26 (verify reconciliation
    with C1 / D4: do the preloaded models register with `ModelManager`, or
    do they still pin to `self`?)*
  - [x] #12 TripoSG flash flag audit — `flash_octree_depth` now only
    passed when `use_flash_decoder=True` (diso available)
  - [x] #21 disk-space pre-flight — `_check_disk_space()` wired into all
    heavy-node load paths
  - [ ] #25a-#25f + D12 canonical `Model3DRef` contract + export metadata
  - [x] Verify `PYTORCH_CUDA_ALLOC_CONF` rollout — all heavy nodes
    (Hunyuan3D, SF3D, TripoSR, Trellis2, TripoSG) now set it

- [ ] **PR-3a "Split + metadata cleanup"** *(high blast radius, but single-repo first)*
  - **`nodetool-huggingface` only**
  - #1 + D2 module split: `local_3d.py` → `text_to_3d.py` + `image_to_3d.py`
    + `_3d_common.py`
  - [x] #8a — done: demoted `hy3dgen`, `pymeshlab`, `scikit-image` to optional
  - #8b, #8e, ~~#8f~~ (done), #8h, plus G4 + G5 audits
  - [x] #20 + D11 license warnings — `license_warning: ClassVar` on each node
  - [x] #22a static VRAM budget class attrs — `MIN_VRAM_GB`, `ESTIMATED_DOWNLOAD_GB`
  - G1 regenerate `package_metadata` after the split

- [ ] **PR-3b "Install UX + runtime hints"** *(cross-repo, product-facing but no new sidebar UI)*
  - **`nodetool-huggingface`**: #8c, #8d installer inputs / pinned requirements
  - **`nodetool`**: `electron/src/packageManager.ts` install actions for #8c
  - **`nodetool-huggingface` / shared runtime path**: #22b-#22c runtime VRAM
    warning via node logs using the existing hardware probe
  - N1 confirm large-GLB delivery path
  - N3 confirm asset-store dedup

- [ ] **PR-3c "Quality + low-VRAM rollout"** *(3D-specific product hardening)*
  - #7 + G7 Hunyuan3D `low_vram_mode` hardening + version pin
  - #18 + D7 `low_vram_mode` rollout (Trellis2 confirmed no-op)
  - #9 OBJ default fix — *PR #26 added the docstring hint; the actual
    `default=GLB` switch on Hunyuan3D / SF3D / TripoSR is still pending,
    verify and finish.* See D1.
  - #25g manual export/viewer verification across generators
  - #26 + D13 manual evaluation matrix for defaults

- [ ] **PR-5 "Generic Model3D post-processing follow-up"** *(separate follow-up track in `nodetool`, not part of the core local-HF 3D fixes)*
  - #28 audit existing `nodetool.model3d.*` nodes for real mesh behavior vs
    heuristic / passthrough behavior
  - #29 upgrade the high-value existing nodes first: `Decimate`,
    `FormatConverter`, `GetModel3DMetadata`, `MergeMeshes`, `Boolean3D`
  - #30 add `NormalizeModel3D` (axis/origin/scale normalization + optional
    ground-plane placement)
  - #31 add `ExtractLargestComponent` / `RemoveSmallComponents`
  - #32 add conservative `RepairMesh`
  - #33 keep UV unwrap / texture-bake / retopology as a later, heavier phase

- [ ] **PR-4 "Apple experimental"** *(separate follow-up PR, after PR-2 or later)*
  - #8i soften SF3D / TripoSR CUDA gates
  - D8 docstring updates marking experimental on macOS
  - Ship as separate PR so it can be reverted independently if it breaks for
    Linux/Windows users.

---

## Affected repos (revised)

- **`nodetool-huggingface`** — the core of the work: Python changes,
  `pyproject.toml`, vendored docs, smoke tests, metadata regeneration.
- **`nodetool`** — needed for install UX and shared runtime plumbing:
  - `electron/src/packageManager.ts` for #8c install actions.
  - `packages/base-nodes/src/nodes/model3d.ts` for the separate generic
    post-processing follow-up track (audit / upgrades / missing nodes).
- **`nodetool-core` / `nodetool-sdk`** — **not required for the current 3D
  plan after dropping backward-compat aliasing and build-time `is_available()`
  ideas.** They may still be part of the general local-HF execution work at
  the end of this file.

---

## Generic Model3D post-processing follow-up

This is a **separate follow-up track**, not part of the core local-HF 3D
generation fixes above.

Important context: nodetool already has a generic `Model3D` node surface in
`nodetool/packages/base-nodes/src/nodes/model3d.ts` with nodes like
`Decimate`, `Transform3D`, `CenterMesh`, `RecalculateNormals`,
`FlipNormals`, `MergeMeshes`, `Boolean3D`, `FormatConverter`, and
`GetModel3DMetadata`. The plan should build on that surface rather than talk as
if post-processing starts from zero.

- [ ] **#28 – Audit existing `nodetool.model3d.*` nodes for real mesh behavior**
  Verify which nodes are already genuinely mesh-aware and which are currently
  convenience-level / heuristic / passthrough implementations. Highest-value
  audit targets: `Decimate`, `FormatConverter`, `GetModel3DMetadata`,
  `MergeMeshes`, `Boolean3D`.

- [ ] **#29 – Upgrade the highest-value existing nodes first**
  Before adding many new nodes, make the obvious ones trustworthy:
  - `Decimate` should do real polygon reduction, not byte-level shrinking
  - `FormatConverter` should perform real mesh conversion
  - `GetModel3DMetadata` should compute true mesh stats
  - `MergeMeshes` / `Boolean3D` should operate on geometry, not raw bytes
  If a node stays heuristic for now, label it clearly rather than silently
  pretending it is production-grade.

- [ ] **#30 – Add `NormalizeModel3D`**
  Add a single convenience node for the most common cleanup step after AI
  generation:
  - normalize axis / orientation
  - center at origin
  - optional uniform scale-to-box
  - optional "rest on ground plane"
  This complements #25 / D12 by giving workflows an explicit postprocess node
  when the upstream model does not export exactly how the user wants.

- [ ] **#31 – Add `ExtractLargestComponent` / `RemoveSmallComponents`**
  AI-generated meshes often include floaters, disconnected islands, and tiny
  junk geometry. A component-cleanup node is one of the highest-value additions
  for actual workflow quality.

- [ ] **#32 – Add a conservative `RepairMesh` node**
  Focus on reliable cleanup:
  - remove degenerate faces
  - merge duplicate / near-duplicate vertices
  - fix simple non-manifold / winding issues when safe
  - report watertightness / repair results in metadata
  Keep scope conservative; do not promise CAD-grade healing in v1.

- [ ] **#33 – Keep UV unwrap / texture-bake / retopology as a later phase**
  These are still recommended eventually, but they are much heavier and more
  failure-prone than the cleanup nodes above. Do not block the local 3D
  generation plan on them.

---

## Newer developments worth considering (April 2026 audit)

### Models we should evaluate

- [ ] **Hunyuan3D-2.1 (Tencent, June 2025) — recommended upgrade**
  Repo: `tencent/Hunyuan3D-2.1`. Two-stage Shape + Paint with **open-source PBR
  texture model** (replacing 2.0's RGB-only). CLIP-FiD 24.78 vs 26.44 for 2.0.
  Full pipeline ~29 GB VRAM but shape-only path stays comparable to 2.0.
  Action: bump `Hunyuan3D` to load `tencent/Hunyuan3D-2.1` shape model; add
  optional Paint stage as a second node (`Hunyuan3DPaint`). Fits in PR-3 or
  a follow-up. **Note: Tencent License remains non-commercial — D11 still
  applies.**

- [ ] **Hunyuan3D-2.5 (Tencent, April 2025) — investigate weight availability**
  10 B-param LATTICE shape model, 4K PBR, 1024 effective geometric resolution.
  Significant quality jump per Tencent's metrics. **Open question:** are the
  weights actually published on HF, or API-only? If open, it's worth a
  separate `Hunyuan3DLattice` node (likely 80 GB VRAM full, smaller variants
  unclear). Verify before committing.

- [ ] **PartCrafter (NeurIPS 2025) — new paradigm node**
  Repo: `wgsxm/PartCrafter`. Single image → **4–16 separate editable part
  meshes** in ~30 s. Compositional latent diffusion transformers. Different
  output type than everything we have today (multi-mesh dict vs single
  `Model3DRef`). Would need a new `Model3DPartListRef` output type or a
  per-part loop. High value for game/CAD workflows. **Suggest: ship as a
  separate PR after PR-3, with a small `nodetool-core` type addition.**

- [ ] **Direct3D-S2 (NeurIPS 2025, May 2025) — future addition**
  Repo: `DreamTechAI/Direct3D-S2`. Sparse SDF VAE + Spatial Sparse Attention,
  1024³ resolution training on 8 GPUs. State-of-the-art for fine geometric
  detail. Larger and slower than TripoSG; probably worth waiting for upstream
  to ship a smaller variant before adding a node. **Track but defer.**

- [ ] **ReLi3D (Stability AI) — multi-view variant of SF3D**
  Repo: `StabilityLabs/ReLi3D`. Takes multi-view images with known camera
  poses → relightable PBR mesh + illumination separation. Different input
  shape (list of `ImageRef` + camera matrices) than current SF3D node.
  Same Stability Community License. **Suggest: separate node if/when there's
  user demand for multi-view; not critical.**

- [ ] **MeshAnything V2 (ICCV 2025) — post-processing node, not generation**
  Repo: `buaacyw/MeshAnythingV2`. Takes any dense mesh or point cloud and
  retopologizes to artist-style quad/tri mesh. Pair with Hunyuan3D / TripoSG
  output as an optional cleanup stage. Fits the "pluggable mesh
  post-processing graph" item already listed under Out of Scope. **Track
  separately.**

- [ ] **3D Gaussian Splatting outputs**
  Trellis2's pipeline already produces gaussian + radiance-field formats
  alongside mesh; we discard them. If the UI ever supports `.splat` / `.ply`
  Gaussian Splat preview, exposing those as separate output formats becomes
  free. **Defer until viewer support exists.**

### Library version bumps

- [ ] **PyTorch 2.10 (released January 2026)**
  We pin `torch==2.9.0`. 2.10 brings Python 3.14 support, `varlen_attn` for
  packed sequences (relevant for the DiT pipelines in Hunyuan3D / TripoSG),
  reduced kernel launch overhead, FP8 on Intel GPUs. **Suggest: bump in a
  dedicated PR (sibling of these), validate the whole HF stack first since
  diffusers 0.36 has had breakage with newer torch.**

- [ ] **diffusers — investigate latest stable**
  We pin `>=0.35.1`. 0.36 had known issues with Flux/WAN; check for a
  patched release before bumping. Shap-E pipeline has had no major changes,
  so low urgency for the 3D nodes specifically.

- [ ] **transformers 5.0 (RC) — defer**
  Major version with breaking changes. Stay on `>=4.56.0` until 5.0 GAs and
  the broader HF ecosystem catches up.

- [ ] **`hy3dgen` — re-pin for Hunyuan3D-2.1**
  If we adopt Hunyuan3D-2.1, the `hy3dgen` package needs a newer pin (or
  switch to `hy3dgen2.1` if Tencent split it). Coordinate with G7.

### What we're NOT missing

Audit-confirmed: Shap-E (OpenAI) has had no significant updates since 2023 —
diffusers pipeline is the canonical open implementation. TripoSR (VAST/Stability,
Feb 2024) is superseded by TripoSG which we already have. Trellis original
(Microsoft 2024) is superseded by Trellis2-4B which we already have.

---

## Out of scope (for a separate effort)

- Adding new local pipelines (Zero123 / Stable Zero123 / InstantMesh / CRM /
  DreamGaussian). Not implemented anywhere in the four repos today.
- Texture-bake post-processing for Hunyuan3D shape-only output (would
  require pulling in `hy3dgen.texgen` + its CUDA C++ extensions).
- A **full** pluggable mesh post-processing graph with advanced operations like
  UV unwrap, retopology, and texture baking. Targeted generic cleanup nodes are
  now tracked in the "Generic Model3D post-processing follow-up" section above.
- OBJ/PLY loaders in `Model3DViewer.tsx` (revisit if user demand surfaces).

---

## General local HuggingFace execution improvements (separate track, not 3D-specific)

These items apply to local Hugging Face execution across nodetool, not just the
3D nodes above. Keep them out of the 3D PR slicing unless we explicitly decide
to broaden scope.

- [ ] **GHF1 – Split static node metadata from live runtime availability**
  Package metadata can safely ship static facts like `SUPPORTED_PLATFORMS`,
  `INSTALL_HINT`, `license_warning`, and `MIN_VRAM_GB`. It should not encode
  live machine state such as "dependency installed", "GPU present", or "GPU has
  enough VRAM". If the app wants those answers, it should query the local
  Python runtime at runtime.

- [ ] **GHF2 – Progress lifecycle for long local inference jobs**
  Standardize high-level stages like: `checking dependencies`,
  `downloading weights`, `loading pipeline`, `preprocessing input`,
  `running inference`, `post-processing output`, `exporting asset`. Long local
  model runs feel broken without visible stage changes.

- [ ] **GHF3 – Cancellation and cleanup semantics**
  Define what happens when the user cancels a local HF job mid-download or
  mid-inference. We need predictable cleanup for partial downloads, temp files,
  and GPU memory so the next run starts from a healthy state.

- [ ] **GHF4 – Better error taxonomy for local inference**
  Distinguish at least: unsupported platform, missing dependency, model download
  failure, out-of-disk, out-of-VRAM, user cancellation, and upstream pipeline
  crash. Raw tracebacks are useful for logs, but the UI should surface concise
  human-readable categories first.

- [ ] **GHF5 – Warm/cold start visibility**
  When preload succeeds, show that a model is warm; when a run pays full load
  cost, show that it is a cold start. This matters for heavyweight local
  inference where "nothing is happening" can just mean "loading 10 GB of
  weights".

- [ ] **GHF6 – CPU-vs-CUDA execution pool strategy**
  `huggingface_pipeline.py` currently uses a global
  `ThreadPoolExecutor(max_workers=1)`, which is reasonable for CUDA memory-pool
  stability but serializes unrelated local HF work too. Add a clearly-scoped
  follow-up design for when CPU-safe nodes should opt out to a separate worker
  path without destabilizing GPU inference.

Possible future product UX, if these warnings prove broadly useful:
- A shared install/runtime status surface (panel, inspector section, or similar)
  for local model availability, license notices, and VRAM guidance across all
  local-HF nodes. This is a roadmap idea only, not a checkable item in this
  plan.
