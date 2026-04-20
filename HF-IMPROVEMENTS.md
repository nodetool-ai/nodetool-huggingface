
## General local HuggingFace execution improvements


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

## 3D-specific ideas moved out of the active plan

These came from the local 3D plan, but are not needed in the current
`FIX-MODEL3D.md` execution order.

- [x] **3D-A1 - Static capability metadata for local 3D nodes** (`#8e`)
  Add static facts like `SUPPORTED_PLATFORMS`, `INSTALL_HINT`,
  `license_warning`, and `MIN_VRAM_GB` where useful for shared product
  surfaces. Keep this separate from live machine-state checks.

- [ ] **3D-A2 - Runtime installer UX for git-only 3D stacks** (`#8c`)
  *Status: deferred / may be dropped.*
  `StableFast3D`, `TripoSR`, and `Trellis2` cannot be installed via `pip` from
  PyPI — they pull C++/CUDA extensions from upstream git repos and need a
  build toolchain (compiler, matching CUDA toolkit, sometimes ninja). A
  proper non-CLI installer would have to ship a build environment, stream
  pip progress through the GHF2 stage channel, map pip failures to the GHF4
  error taxonomy, and refuse early on unsupported platforms. That is a
  significant effort for three nodes whose audience is mostly
  CLI-comfortable.

  **For now:** leave the nodes registered as-is. Each node's `INSTALL_HINT`
  must make the manual-effort cost obvious — point users at the pinned
  `requirements/<node>.txt` file, mention the build toolchain / CUDA
  requirement, and call out the platform constraints. PyPI-installable
  nodes (Shap-E, Hunyuan3D, TripoSG) are unaffected and continue to use the
  normal `pip install 'nodetool-huggingface[<extra>]'` path.

  Revisit only if these three nodes get enough non-CLI demand to justify
  the installer infrastructure; otherwise consider dropping them in favor
  of provider-based 3D generation (Meshy / Rodin) which is in
  `nodetool` core.

- [x] **3D-A3 - VRAM guidance / warning path for 3D nodes** (`#22b`, `#22c`)
  Surface soft warnings when the detected machine is below a model's
  recommended VRAM, plus add a focused test and one manual verification note.

- [ ] **3D-A4 - Large-asset delivery path for generated 3D outputs** (`N1`)
  Confirm large GLBs route through asset storage rather than inline WebSocket
  frames.

- [ ] **3D-A5 - Asset-store uniqueness for repeated 3D node runs** (`N3`)
  Confirm repeated `Model3DRef` outputs do not overwrite each other.

- [ ] **3D-A6 - Manual benchmark matrix for local 3D defaults** (`#26`)
  Keep a lightweight runtime / VRAM / viewer-success benchmark sheet for a few
  representative prompts or images.

- [ ] **3D-A7 - Make the phase-1 product boundary explicit** (`#27`)
  Keep a short statement of what the local 3D effort does and does not include,
  without making it part of the active implementation queue.
