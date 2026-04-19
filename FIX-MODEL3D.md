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

## Quick-win priority (do these first)

- [ ] **#2 – Delete dead `Trellis2._ensure_model_downloaded`**
  Lines `918–939`. Copy-paste from `Hunyuan3D`, references a non-existent
  `self.VARIANT_CONFIG`. Currently unreachable but will `AttributeError` if any
  refactor wires it up. Just delete it.

- [ ] **#3 – Override `requires_gpu() -> False` on both Shap-E nodes**
  Shap-E's `process()` already supports CPU (lines 98, 227), but the base
  class default `requires_gpu() -> True` (huggingface_pipeline.py line 171)
  prevents the scheduler from running these on CPU/MPS-only machines.

- [ ] **#10 – Cache `rembg` session in `ModelManager`**
  `SF3D` (line 661) and `TripoSR` (line 800) call `rembg.new_session()` per
  invocation, re-loading U²-Net every time. Cache once with key
  `"rembg_u2net_session"`.

- [ ] **#13 – Standardize seeding to per-call `torch.Generator`**
  Inconsistent today:
  - Shap-E: per-call `torch.Generator` (good).
  - Hunyuan3D / Trellis2: global `torch.manual_seed(...)` (leaks state across
    runs, lines 495, 988).
  - TripoSG: per-call `torch.Generator` (good).
  Pick per-call generators everywhere. See D3.

- [ ] **#15 – Add `preload_model` to the 5 heavy nodes**
  Only `ShapETextTo3D` and `ShapEImageTo3D` implement `preload_model`.
  `Hunyuan3D`, `StableFast3D`, `TripoSR`, `Trellis2`, `TripoSG` all load
  lazily inside `process()`, so the executor's preload phase is a no-op for
  them and the first run pays full cold-start. **See C1 — must be reconciled
  with D4.**

- [ ] **#9 – Resolve viewer/output format mismatch**
  `Model3DViewer.tsx` (lines 551–558) only renders GLB/GLTF, but Hunyuan3D /
  SF3D / TripoSR expose an OBJ output option. Either restrict outputs to GLB
  (simpler) or add `OBJLoader` / `PLYLoader` to the viewer (richer).
  See Decisions §D1.

- [ ] **#5 – Extract a single `_export_mesh(mesh, format) -> bytes` helper**
  The "trimesh export, with optional `.cpu().numpy()` fallback" code is
  duplicated 5× with slight per-node variations (`include_normals=True` only
  for SF3D, `extension_webp=True` only for Trellis2). Centralize. **See C2 —
  the o_voxel-direct-to-GLB path stays separate.**

- [ ] **#1 – Split module by input modality (HF `pipeline_tag` aligned)**
  Current `text_to_3d.py` contains 6 image→3D nodes. Split into
  `text_to_3d.py` and `image_to_3d.py`, matching HuggingFace's official
  `pipeline_tag` values (`text-to-3d`, `image-to-3d`). See Decisions §D2.

- [ ] **#16 – Add a no-GPU smoke test**
  At minimum: import the module, instantiate each node, assert
  `get_recommended_models()` returns valid `HuggingFaceModel`s. That single
  test would have caught finding #2.

---

## Dependency strategy (#8 — expanded)

Single biggest user-visible issue: today `pip install nodetool-huggingface`
declares `hy3dgen`, `pymeshlab`, `scikit-image` as **hard** dependencies, while
`sf3d`, `tsr`, `trellis2`, `o_voxel`, `rembg`, `diso` are not declared at all
and `ImportError` at runtime. We split this into A+C+D:

- [ ] **#8a – Demote heavy deps from hard to optional**
  Move `hy3dgen`, `pymeshlab`, `scikit-image` out of `[project] dependencies`.
  Goal: `pip install nodetool-huggingface` succeeds on a clean macOS box.

- [ ] **#8b – Adopt env-markered `[project.optional-dependencies]` (Option A)**
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
  `uv pip install git+https://...@<commit>` with platform gating. The node
  sidebar shows an "Install" button instead of a stack trace.
  Stacks: `sf3d`, `triposr`, `trellis2`.

- [ ] **#8d – Pin upstream commits for git-only deps (Option D)**
  Maintain `dev/requirements-3d-sf3d.txt` / `-triposr.txt` / `-trellis2.txt`
  with commit-pinned VCS URLs for CI / dev reproducibility. The runtime
  installer (#8c) reads these.

- [ ] **#8e – `is_available()` per node, with platform + import check**
  ```python
  @classmethod
  def is_available(cls) -> bool:
      if platform.system() != "Linux":
          return False  # Trellis2 only
      try:
          import trellis2  # noqa
          return True
      except ImportError:
          return False
  ```
  Sidebar message reflects which condition failed: "Linux only" vs
  "Install with `nodetool install-extra trellis2`".

- [ ] **#8f – `**Platforms:**` line in every heavy-node docstring**
  - Shap-E: *"Platforms: all (CPU/MPS/CUDA)."*
  - Hunyuan3D: *"Platforms: Linux+CUDA, Windows+CUDA. Installable on macOS but does not run."*
  - SF3D: *"Platforms: Linux+CUDA, Windows+CUDA. macOS Metal experimental (untested)."*
  - TripoSR: *"Platforms: Linux+CUDA, Windows+CUDA. macOS CPU experimental (slow, untested)."*
  - TripoSG: *"Platforms: Linux+CUDA, Windows+CUDA (build required)."*
  - Trellis2: *"Platforms: Linux+CUDA only, 24 GB+ VRAM."*

- [ ] **#8i – Soften CUDA gates in SF3D and TripoSR**
  Today both raise `RuntimeError("requires CUDA")` (lines 638-640, and TripoSR
  similarly) before the upstream code even gets a chance. Replace with a
  permissive device check + warning when running on non-CUDA, and let the
  underlying library fail with its own (more informative) error. This unlocks
  the experimental Apple paths above.

- [ ] **#8g – Skip Option B (direct VCS deps in `pyproject.toml`)**
  PyPI rejects wheels with VCS deps in metadata; would block our publish
  pipeline. Documented for posterity.

- [ ] **#8h – Validate combined install resolves**
  Confirm `[hunyuan3d,triposg]` resolves with our pinned `torch==2.9.0`. If
  not, document mutually-exclusive groups.

---

## Correctness fixes

- [ ] **#4 – Shap-E ignores the device chosen by `_resolve_hf_device`**
  `load_model` may place weights on `mps` (preferred on Apple Silicon by
  `local_provider_utils._resolve_hf_device`), but Shap-E's `process()` re-does
  `"cuda" if torch.cuda.is_available() else "cpu"` at lines 98, 114, 227,
  249. On macOS the `torch.Generator` ends up on `cpu` while the model is on
  `mps`. Drive everything off `self._pipeline.device`.

- [ ] **#6 – Stop holding strong refs on the node instance**
  Pattern `self._model = ModelManager.get_model(cache_key) or new_model;
  ModelManager.set_model(...)` keeps a strong reference on `self._model`. If
  `ModelManager` evicts under VRAM pressure, the GPU memory still can't be
  freed because the node instance pins it. Always read through
  `ModelManager.get_model(cache_key)` and don't stash on `self`.
  **See G6 — reconcile with `move_to_device`.**

- [ ] **#7 – Hunyuan3D `low_vram_mode` monkey-patch may break silently**
  Lines 476–487 fake a `.components` dict on the `hy3dgen` pipeline before
  calling `enable_model_cpu_offload()`. That helper relies on more than just
  `.components` (hooks, `_execution_device`, `_offload_gpu_id`). Wrap in a
  try/except with a clear "low_vram_mode unavailable for this hy3dgen
  version" warning, and pin the validated `hy3dgen` version range (G7).

- [ ] **#11 – TripoSG `_prepare_image` hard-codes `.cuda()`**
  Lines 1181–1234 use literal `.cuda()`. Currently fine because `process()`
  gates on CUDA, but brittle. Use `device` consistently.

- [ ] **#12 – TripoSG always passes `flash_octree_depth`**
  Even when `use_flash_decoder=False` (line 1349). Verify upstream behavior;
  at depth 9 this can blow VRAM if the pipeline preallocates flash buffers
  unconditionally.

- [ ] **#14 – `seed=-1` semantics inconsistent**
  Most nodes treat `-1` as "no generator". TripoSG defaults to `42` and
  converts `-1` to a freshly random seed it never echoes back. Pick one
  convention; if random is allowed, surface the actually-used seed in
  output metadata for reproducibility. See Decisions §D3.

---

## Polish / UX

- [ ] **#17 – Document vendored `triposg` upstream**
  `src/triposg/` is excluded from ruff but ships in the wheel. Add
  `src/triposg/UPSTREAM.md` with the upstream commit SHA and any local
  patches so future merges are tractable.

- [ ] **#18 – Add `low_vram_mode` to all heavy nodes**
  Currently only `Hunyuan3D` exposes it. Trellis2 (24 GB), SF3D (~6 GB), and
  TripoSG (~8 GB) could expose `enable_sequential_cpu_offload()` /
  `enable_model_cpu_offload()` behind the same flag. See D7.

- [ ] Hunyuan3D `_ensure_model_downloaded` docstring still says "75 GB repo"
  (line 412). Make sure the message matches reality after the
  `allow_patterns` fix.

- [ ] Verify all heavy nodes set
  `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")`
  the way Trellis2 does (line 948) — fragmentation hurts every long-lived
  CUDA worker, not just Trellis2.

---

## Gaps surfaced during planning

- [ ] **G1 – Regenerate `package_metadata/nodetool-huggingface.json`**
  After #1 rename and #8 dependency / `is_available` changes, the package
  metadata JSON must be regenerated. Run the regen script and verify only
  intended diffs.

- [x] **G2 – Migration shim for existing workflows referencing `huggingface.text_to_3d.<image-node>`**
  **Verified: `get_node_class()` in `nodetool-core/src/nodetool/workflows/base_node.py:2299`
  does a direct `NODE_BY_TYPE` dict lookup with NO alias support.** Adding
  aliasing requires a `nodetool-core` change. **D2 is a 2-repo change**:
  - In `nodetool-core`: add `NODE_TYPE_ALIASES: dict[str, str]` and check it
    in `_lookup()` before returning `None`. ~10 lines.
  - In `nodetool-huggingface`: register six per-node aliases at import time
    (one for each node moving from `text_to_3d.py` to `image_to_3d.py`):
    `huggingface.text_to_3d.ShapEImageTo3D`   → `huggingface.image_to_3d.ShapEImageTo3D`
    `huggingface.text_to_3d.Hunyuan3D`        → `huggingface.image_to_3d.Hunyuan3D`
    `huggingface.text_to_3d.StableFast3D`     → `huggingface.image_to_3d.StableFast3D`
    `huggingface.text_to_3d.TripoSR`          → `huggingface.image_to_3d.TripoSR`
    `huggingface.text_to_3d.Trellis2`         → `huggingface.image_to_3d.Trellis2`
    `huggingface.text_to_3d.TripoSG`          → `huggingface.image_to_3d.TripoSG`
  - `ShapETextTo3D` namespace is unchanged → no alias needed.

- [x] **G3 – Confirm `is_available()` is a real `BaseNode` hook**
  **Verified: `BaseNode` has `is_visible()` (line 624) and `requires_gpu()`
  (line 2009) but NO `is_available()`.** Adding it is a small core change
  that follows the established pattern.
  - `nodetool-core`: add `@classmethod def is_available(cls) -> tuple[bool, str | None]`
    returning `(available, reason_if_not)`. Default `(True, None)`.
  - `nodetool-sdk/csharp/Nodetool.Types/scripts/generation/discovery.py` line 201
    already filters by `is_visible()` — extend to also surface
    `is_available()` into the generated metadata.
  - `nodetool` web sidebar consumes the metadata and renders the warning.
  - **Decision: do this in `nodetool-core`** rather than the fallback "raise
    in pre_process" approach. Cost is small, UX is much better.

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

- [ ] **G7 – Pin `hy3dgen` version**
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

- [ ] **#19 – Pin model revisions in `from_pretrained`**
  Every `from_pretrained` / `snapshot_download` call uses the bare repo id
  with no `revision=` pin. If upstream pushes breaking changes (config rename,
  weight reorganization), our nodes silently break for users. Pin a known-good
  commit SHA per model; bump intentionally. Affects Shap-E (lines 100-105,
  229-234), Hunyuan3D (line 423), SF3D (line 651), TripoSR (line 791),
  Trellis2 (line 980), TripoSG (lines 1304, 1318).

- [ ] **#20 – SF3D licensing surfaced in node UI**
  Stable Fast 3D is **Stability AI Community License** — free under
  $1M revenue, enterprise license required above. Today buried in the
  docstring (line 558). Surface as a `license_warning` field that the
  sidebar displays when the node is added.

- [ ] **#21 – Disk-space pre-flight for model downloads**
  Trellis2 = 10 GB+, Hunyuan3D standard = 5 GB, TripoSG = 3 GB. A user on a
  small SSD can hit "no space left on device" mid-download with a corrupt
  cache. Add a pre-flight check via `shutil.disk_usage()` against the HF
  cache directory before `snapshot_download`, with a clear error.

- [ ] **#22 – Document VRAM budgets in node metadata, not just docstrings**
  Each heavy node knows its own VRAM minimum (Hunyuan3D 6 GB, SF3D 6 GB,
  TripoSR 6 GB, TripoSG 8 GB, Trellis2 24 GB). Expose this as a class
  attribute (e.g. `MIN_VRAM_GB: ClassVar[int] = 24`) so the scheduler /
  sidebar can warn users *before* OOM. Fold into the `is_available()` check
  via `torch.cuda.get_device_properties(0).total_memory`.

- [ ] **#23 – `_pipeline_thread_pool` is a global single worker**
  `huggingface_pipeline.py` line 25: `ThreadPoolExecutor(max_workers=1)`
  serializes ALL HF inference. For 3D specifically that's largely fine (one
  Trellis2 job at 60s on H100 saturates the device anyway), but unrelated
  audio / Shap-E CPU runs also serialize. **Decision: out of scope for the 3D
  PRs.** Add a docstring comment marking the pool as intentionally
  single-worker for CUDA memory pool consistency, and leave a TODO referencing
  a future `runs_on_cpu_pool: ClassVar[bool] = False` opt-out marker so
  CPU-bound HF nodes can route through `asyncio.to_thread` instead.

- [ ] **#24 – Input image validation**
  Each `process()` calls `await context.asset_to_io(self.image)` then opens
  with PIL. Failure modes (corrupt image, EXIF-rotated, palette-only) all
  bubble as raw `PIL.UnidentifiedImageError`. Wrap once in a helper that
  returns a friendly `ValueError("Invalid input image: <reason>")`.

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

**Migration:**

- The `huggingface.text_to_3d.ShapETextTo3D` namespace is unchanged → no
  alias needed for that one node.
- All six image→3D nodes need an alias from
  `huggingface.text_to_3d.<NodeName>` → `huggingface.image_to_3d.<NodeName>`
  for one release. Drop in the next minor version.
- Use the alias mechanism added per **G2** (`nodetool-core` change).
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
- `is_available()` returns `False` on platform mismatch *and* `ImportError`.
  Sidebar message distinguishes the two cases.
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

**Decision: class attribute + scheduler hint, no hard block.**

- Each heavy node declares `MIN_VRAM_GB: ClassVar[int]`.
- `is_available()` returns `True` even if VRAM is below the budget — but adds
  a warning to the sidebar ("This model needs ~24 GB VRAM, you have 12 GB.
  Likely to OOM. Try Hunyuan3D Mini instead.").
- We don't hard-block because users with `low_vram_mode=True` may make it
  work, and we shouldn't over-promise either way.

### D11 – Licensing surfacing (finding #20)

**Decision: `license_warning: ClassVar[str | None]` on each node, rendered
in the sidebar.**

- Default `None` (no warning) — e.g. Shap-E MIT, TripoSR MIT, TripoSG MIT.
- Set on **Hunyuan3D** (Tencent non-commercial), **SF3D** (Stability
  Community License, $1M revenue cap), **Trellis2** (Microsoft Research
  License, non-commercial) with concise human-readable text + upstream
  license URL.
- **No "I accept" gate** — sidebar warning is sufficient for now. Revisit
  if any user actually trips over the licensing.

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

- [ ] **PR-1 "Quick fixes"** *(low risk, no API changes)*
  - #2 delete dead code
  - #3 Shap-E `requires_gpu`
  - #4 Shap-E device fix
  - #11 TripoSG `_prepare_image` cleanup
  - #19 + D9 pin model revisions (table only, no behavior change)
  - #24 input-image validation helper
  - Hunyuan3D docstring cleanup
  - #16 + N2 smoke test
  - #17 vendored `triposg/UPSTREAM.md`

- [ ] **PR-2 "Refactor & helpers"** *(medium, internal API churn)*
  - #5 + C2 export helper
  - #10 rembg cache
  - #6 + G6 ModelManager refactor
  - #13 + #14 + D3 seeding rework
  - #15 + C1 `preload_model` rollout
  - #12 TripoSG flash flag audit
  - #21 disk-space pre-flight
  - Verify `PYTORCH_CUDA_ALLOC_CONF` rollout

- [ ] **PR-3 "Renames & packaging"** *(high blast radius, ship last; spans 4 repos)*
  - **`nodetool-core` first**: G2 alias mechanism in `get_node_class()`,
    G3 `BaseNode.is_available()` hook.
  - **`nodetool-sdk`**: extend discovery to surface `is_available()`,
    regenerate types.
  - **`nodetool` web**: sidebar warnings consume new metadata
    (#8e + #20/D11 + #22/D10), D3 seed badge in `Model3DProperty.tsx`.
    `electron/src/packageManager.ts` install actions for #8c.
  - **`nodetool-huggingface` last**:
    - #1 + D2 module/namespace rename + alias registration
    - #8a–#8i + G4 + G5 dependency strategy
    - #7 + G7 Hunyuan3D `low_vram_mode` hardening + version pin
    - #18 + D7 `low_vram_mode` rollout (Trellis2 confirmed no-op)
    - #9 + D1 viewer / OBJ default fix
    - #20 + D11 license warnings (data only)
    - #22 + D10 VRAM budget class attrs
    - G1 regenerate package_metadata
    - N1 confirm large-GLB delivery path
    - N3 confirm asset-store dedup

- [ ] **PR-4 (optional) "Apple experimental"** *(can ship anytime after PR-2)*
  - #8i soften SF3D / TripoSR CUDA gates
  - D8 docstring updates marking experimental on macOS
  - Ship as separate PR so it can be reverted independently if it breaks for
    Linux/Windows users.

---

## Affected repos (revised after G2 / G3 verification)

- **`nodetool-huggingface`** — ~80% of the work (all Python changes,
  `pyproject.toml`, vendored docs, smoke tests).
- **`nodetool-core`** — **confirmed required** in PR-3:
  - `NODE_TYPE_ALIASES: dict[str, str]` + alias check in `_lookup()` of
    `get_node_class()` (G2 — verified missing).
  - `@classmethod is_available(cls) -> tuple[bool, str | None]` on
    `BaseNode` returning `(available, reason)`. Default `(True, None)` (G3 —
    verified missing).
- **`nodetool-sdk`** — `csharp/Nodetool.Types/scripts/generation/discovery.py`
  must surface `is_available()` into generated metadata (extends existing
  `is_visible()` filter at line 201). Auto-regenerated types follow.
- **`nodetool`** — PR-3:
  - `electron/src/packageManager.ts` for #8c install actions.
  - `web/src/components/properties/Model3DProperty.tsx` and sidebar
    component for #8e/#22/#20 warnings (license, VRAM budget, missing deps)
    + D3 resolved-seed badge.

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
- Pluggable mesh post-processing graph (decimation, remesh, UV unwrap) as
  separate nodes that consume `Model3DRef`.
- OBJ/PLY loaders in `Model3DViewer.tsx` (revisit if user demand surfaces).
