# Model3D cross-repo follow-ups

This file tracks `Model3D` work that belongs outside
`nodetool-huggingface`, mainly in the `nodetool` repo.

It is intentionally separate from `FIX-MODEL3D.md` so the HF generator
cleanup can stay top-to-bottom executable without mixing in generic mesh
tooling work.

---

## Scope

Primary target repo: `nodetool`

Not required to finish the active HF 3D fix plan, but worth doing once the
generator layer is stable enough to hand off clean `Model3DRef` outputs.

---

## Existing state already confirmed

Audit result from `nodetool/packages/base-nodes/src/nodes/model3d.ts`:

- `Transform3D`, `CenterMesh`, `RecalculateNormals`, and `FlipNormals`
  already do real mesh work and should be kept.
- `Decimate`, `FormatConverter`, `GetModel3DMetadata`, `Boolean3D`, and
  `MergeMeshes` already exist, but today they are placeholders, heuristics,
  or byte-level hacks and should be upgraded instead of replaced by duplicate
  nodes.
- There is no settled shared library stack yet, so this plan chooses one
  explicitly below instead of leaving each node to invent its own approach.

---

## Chosen library stack

Decision: the first trustworthy `Model3D` upgrade pass should use exactly these
three libraries, with clear ownership boundaries:

- **`glTF-Transform` is the base layer**
  Use it for canonical GLB/GLTF parsing and writing, metadata extraction,
  document-level edits, and transform-safe scene operations. This should back
  `FormatConverter`, `GetModel3DMetadata`, and most non-destructive mesh/scene
  handling.
- **`meshoptimizer` owns simplification**
  Use it for real geometry simplification behind `Decimate`. Do not build a
  custom decimator and do not fake decimation with container-level tricks.
- **`manifold-3d` owns booleans and solid-style union**
  Use it when `Boolean3D` or a union-style `MergeMeshes` needs actual
  geometry-aware behavior instead of triangle concatenation.

Notes:

- Keep **GLB as the canonical transport format** even if some nodes import from
  or export to other formats.
- Do **not** add a fourth general-purpose mesh library in the first pass unless
  one of the planned nodes is still blocked after trying this stack.
- Do not add a larger post-processing stack unless a concrete node still
  cannot be implemented cleanly after trying the three libraries above.
- Treat UV unwrap, retopology, and texture baking as a later phase, not a
  dependency for shipping the first trustworthy cleanup nodes.

---

## Execution order

Treat this as a strict queue. Do not skip ahead.

Rule: do not start the next phase until the current phase is implemented,
tested, and the plan file is updated.

### 0. Establish the base layer first

- [ ] **Install only the first-slice libraries**
  Add `glTF-Transform` and `meshoptimizer` to `nodetool` first.
- [ ] **Do not install `manifold-3d` yet**
  Hold it back until work actually begins on `Boolean3D` or union-style
  `MergeMeshes`.
- [ ] **Create shared helpers before upgrading nodes**
  Add focused internal helpers for GLB loading/writing and metadata extraction
  so `FormatConverter`, `GetModel3DMetadata`, and later nodes do not each build
  their own parsing path.

Exit criteria for phase 0:

- `nodetool` builds with the new first-slice dependencies.
- There is one shared GLB/document helper path ready to reuse.

### 1. Make the existing I/O and metadata nodes honest

Start here. This is the first real implementation slice.

- [ ] **Make `FormatConverter` a real converter**
  Keep GLB as the internal transport format. Replace byte-level or extension
  rename behavior with real parse-and-write flows for the formats we actually
  claim to support.
- [ ] **Make `GetModel3DMetadata` read real geometry**
  Return actual bounds, primitive counts, mesh counts, material presence, and
  other cheap-to-compute facts from parsed model data rather than filename or
  container heuristics.
- [ ] **Add fixture tests for these two nodes immediately**
  Do not defer tests to the end. Add at least one real GLB fixture and assert
  geometry-aware outputs.

Exit criteria for phase 1:

- `FormatConverter` no longer lies by only relabeling bytes.
- `GetModel3DMetadata` reports real geometry-derived values.
- Tests cover both nodes with real fixtures.

### 2. Upgrade simplification

- [ ] **Make `Decimate` perform real simplification**
  Use `glTF-Transform` + `meshoptimizer` so triangle count changes are tied to
  mesh geometry, not placeholder behavior.
- [ ] **Use the wrapper path first**
  Try `NodeIO` + `weld()` + `simplify()` before writing any custom low-level
  simplification plumbing.
- [ ] **Keep v1 GLB-focused**
  Support real GLB decimation first. Do not block this phase on OBJ/STL/PLY.
- [ ] **Document the truthful limits of `Decimate`**
  If only a subset of meshes or formats is supported at first, say so in node
  docs and metadata instead of pretending to support everything.
- [ ] **Add fixture tests for `Decimate`**
  Assert that simplification changes geometry statistics in the expected
  direction, not just that some output bytes exist.

Exit criteria for phase 2:

- `Decimate` uses a real simplification path.
- Supported inputs and limitations are documented.
- Tests verify geometry-level effects.

### 3. Then add boolean / merge work

Only start this phase once phases 0-2 are stable.

- [ ] **Install `manifold-3d`**
  Add it only when work on booleans actually starts.
- [ ] **Make `Boolean3D` geometry-aware**
  Implement union / difference / intersection on real meshes or narrow the
  supported modes until the implementation is truthful.
- [ ] **Make `MergeMeshes` explicit about behavior**
  Support at least one honest merge mode first: either scene merge
  (concatenate while preserving transforms) or boolean union. Do not keep one
  node that claims both if only one path is reliable.
- [ ] **Add fixture tests for both nodes**
  Include success cases and at least one documented unsupported-input case.

Exit criteria for phase 3:

- `Boolean3D` and `MergeMeshes` have truthful behavior.
- Unsupported cases are explicit.
- Tests cover both success and failure/limit paths.

### 4. Add the missing cleanup nodes

- [ ] **Add `NormalizeModel3D`**
  Provide one focused node for centering, orientation normalization, optional
  uniform scale-to-box, and optional ground-plane placement.
- [ ] **Add component cleanup**
  Add `ExtractLargestComponent` and/or `RemoveSmallComponents` so workflows can
  remove floaters and disconnected junk geometry from AI-generated outputs.
- [ ] **Add a conservative `RepairMesh`**
  Start with reliable fixes only: degenerate-face removal, duplicate or
  near-duplicate vertex merging, and similarly safe cleanup passes.
- [ ] **Add tests as each node lands**
  Do not batch all cleanup-node tests into a later PR.

Exit criteria for phase 4:

- Each new cleanup node has one narrow purpose.
- Each new cleanup node ships with focused fixture coverage.

### 5. Final verification pass

- [ ] **Review node docs and titles for honesty**
  If a node only supports triangular meshes, manifold inputs, or a subset of
  formats, state that in metadata and docs instead of implying universal mesh
  support.
- [ ] **Run repo verification**
  Run the relevant `nodetool` build, typecheck, lint, and tests before calling
  the upgrade pass done.

---

## Explicitly later

- [ ] **UV unwrap / texture bake / retopology**
  Keep out of the first cleanup pass.
- [ ] **A broad "do everything" mesh pipeline**
  Prefer small trustworthy nodes over one oversized post-processing node.
- [ ] **MeshAnything V2 or similar post-processing models**
  Revisit only after the generic geometry node layer becomes reliable.

