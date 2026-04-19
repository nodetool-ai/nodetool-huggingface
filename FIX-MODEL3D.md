# Local 3D Generation – Review & Fix Plan

Scope: `src/nodetool/nodes/huggingface/text_to_3d.py` (and the related viewer in
`nodetool/web/src/components/asset_viewer/Model3DViewer.tsx`). Covers Shap-E,
Hunyuan3D, StableFast3D, TripoSR, Trellis2, TripoSG. Does NOT cover fal / kie /
replicate.

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

- [ ] **#8 – Add `[project.optional-dependencies]` groups**
  `pyproject.toml` does NOT declare `sf3d`, `tsr`, `trellis2`, `o_voxel`,
  `rembg`, `diso`. Each raises `ImportError` at runtime today. Add e.g.:
  ```toml
  [project.optional-dependencies]
  sf3d = ["sf3d", "rembg"]
  triposr = ["tsr", "rembg"]
  trellis2 = ["trellis2", "o_voxel"]
  triposg = ["diso"]   # plus rembg if not vendored
  all-3d = ["nodetool-huggingface[sf3d,triposr,trellis2,triposg]"]
  ```
  Plus a clear "missing dependency" UI hint or `is_available()` gating.

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
  Pick per-call generators everywhere.

- [ ] **#15 – Add `preload_model` to the 5 heavy nodes**
  Only `ShapETextTo3D` and `ShapEImageTo3D` implement `preload_model`.
  `Hunyuan3D`, `StableFast3D`, `TripoSR`, `Trellis2`, `TripoSG` all load
  lazily inside `process()`, so the executor's preload phase is a no-op for
  them and the first run pays full cold-start.

- [ ] **#9 – Resolve viewer/output format mismatch**
  `Model3DViewer.tsx` (lines 551–558) only renders GLB/GLTF, but Hunyuan3D /
  SF3D / TripoSR expose an OBJ output option. Either restrict outputs to GLB
  (simpler) or add `OBJLoader` / `PLYLoader` to the viewer (richer).
  See Decisions §D1.

- [ ] **#5 – Extract a single `_export_mesh(mesh, format) -> bytes` helper**
  The "trimesh export, with optional `.cpu().numpy()` fallback" code is
  duplicated 5× with slight per-node variations (`include_normals=True` only
  for SF3D, `extension_webp=True` only for Trellis2). Centralize.

- [ ] **#1 – Rename module/namespace away from `text_to_3d`**
  Module name and node namespace `huggingface.text_to_3d.*` are misleading
  since 6 of 7 nodes are image→3D. See Decisions §D2.

- [ ] **#16 – Add a no-GPU smoke test**
  At minimum: import the module, instantiate each node, assert
  `get_recommended_models()` returns valid `HuggingFaceModel`s. That single
  test would have caught finding #2.

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

- [ ] **#7 – Hunyuan3D `low_vram_mode` monkey-patch may break silently**
  Lines 476–487 fake a `.components` dict on the `hy3dgen` pipeline before
  calling `enable_model_cpu_offload()`. That helper relies on more than just
  `.components` (hooks, `_execution_device`, `_offload_gpu_id`). Wrap in a
  try/except with a clear "low_vram_mode unavailable for this hy3dgen
  version" warning, and pin the validated `hy3dgen` version range.

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
  `enable_model_cpu_offload()` behind the same flag.

- [ ] Hunyuan3D `_ensure_model_downloaded` docstring still says "75 GB repo"
  (line 412). Make sure the message matches reality after the
  `allow_patterns` fix.

- [ ] Verify all heavy nodes set
  `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")`
  the way Trellis2 does (line 948) — fragmentation hurts every long-lived
  CUDA worker, not just Trellis2.

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

### D2 – Module / namespace rename (finding #1)

**Decision: rename file to `local_3d.py`, namespace to `huggingface.local_3d.*`.**

- Single file is fine — splitting into `text_to_3d.py` + `image_to_3d.py`
  duplicates imports and helpers, and `ShapETextTo3D` is the only true
  text→3D node.
- Keep an alias in node-type metadata (`huggingface.text_to_3d.*` →
  `huggingface.local_3d.*`) for one release so existing workflows keep
  loading. Drop the alias in the next minor version.

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

- Nodes hold a *weak* convenience handle (`self._model = None` after each
  `process()` exits) and always re-fetch via `ModelManager.get_model(key)` at
  the top of `process()`.
- Shared subsystems get their own keys: `"rembg_u2net_session"`,
  `"briaai/RMBG-1.4_BriaRMBG"`, etc.

### D5 – Optional dependencies packaging (finding #8)

**Decision: declare `[project.optional-dependencies]` groups + soft-fail in the UI.**

- Groups: `sf3d`, `triposr`, `trellis2`, `triposg`, `all-3d`.
- Each node implements a `@classmethod is_available()` that imports the heavy
  module and returns `False` on `ImportError`. Unavailable nodes are still
  visible (so users discover them) but show a "missing dependency: install
  with `pip install nodetool-huggingface[trellis2]`" hint in the sidebar
  instead of failing at queue time.

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
"manual smoke runs".

### D7 – `low_vram_mode` rollout (finding #18)

**Decision: ship behind a single shared `low_vram_mode` field on each heavy
node, but implement per-node:**

- `Hunyuan3D`: keep current monkey-patched `enable_model_cpu_offload` (with
  try/except hardening from #7).
- `SF3D`, `TripoSG`: call `enable_model_cpu_offload()` if the pipeline
  supports it, else no-op with a log warning.
- `Trellis2`: leave as not-supported for now (24 GB minimum is the
  contract); revisit if upstream adds offload.

---

## Out of scope (for a separate effort)

- Adding new local pipelines (Zero123 / Stable Zero123 / InstantMesh / CRM /
  DreamGaussian). Not implemented anywhere in the four repos today.
- Texture-bake post-processing for Hunyuan3D shape-only output.
- Pluggable mesh post-processing graph (decimation, remesh, UV unwrap) as
  separate nodes that consume `Model3DRef`.
