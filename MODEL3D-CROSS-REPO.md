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

## Recommended later libraries

These are worth evaluating after the first cleanup pass is stable. They are not
part of the initial three-library stack above and should not be pulled in early
without a concrete consumer.

- **Add `@gltf-transform/functions` and `@gltf-transform/extensions` before
  reaching for another general-purpose scene library**
  Treat these as the natural next step on top of the chosen `glTF-Transform`
  base layer. Prefer existing document helpers such as `NodeIO`, `weld()`,
  `prune()`, `dedup()`, `reorder()`, and extension-aware material / texture
  handling before inventing custom GLB plumbing.
- **`three-mesh-bvh` is the first scene-preview helper worth adding**
  Use it only when larger scene preview work needs better picking, raycasting,
  or spatial-query performance. It is a viewer / interaction helper, not a
  replacement for geometry processing.
- **Keep gaussian splat support on a separate track**
  If we want frontend preview of splats, prefer a dedicated viewer library such
  as `@mkkellogg/gaussian-splats-3d` in the existing Three.js ecosystem, or a
  Babylon.js-based preview path if Babylon becomes useful for richer inspection.
  Do not force splats through mesh cleanup nodes.
- **Treat Babylon.js as preview / inspection infrastructure, not the workflow
  conversion backbone**
  Babylon may be useful later for broader file preview, scene inspection,
  lighting, PBR checks, and optional browser-side GLB export. It should not
  displace `glTF-Transform` as the canonical document layer for `FormatConverter`
  or metadata nodes.
- **Consider `loaders.gl` only if broader import coverage becomes necessary**
  This is an ingest helper candidate for formats we do not want to parse
  ourselves. If adopted, normalize into GLB quickly and keep workflow operations
  on the canonical GLB path.
- **Defer texture / material pipeline additions until scene or PBR work proves
  the need**
  Good later candidates include KTX2 / Basis texture tooling and `xatlas` /
  `xatlas-wasm` for UV unwrap, but neither should block the first trustworthy
  mesh-node pass.

Rule of thumb:

- Preview breadth can use specialized viewer libraries.
- Workflow nodes should still normalize into GLB and operate on honest,
  geometry-aware document data.

---

## Execution order

Treat this as a strict queue. Do not skip ahead.

Rule: do not start the next phase until the current phase is implemented,
tested, and the plan file is updated.

Current state:

- Phases 0-4 are now landed in `nodetool` on the active branch.
- The next active step is to add the missing cleanup nodes on top of the new
  `model3d/` module layout.

### 0. Establish the base layer first

- [x] **Install only the first-slice libraries**
  Add `glTF-Transform` and `meshoptimizer` to `nodetool` first.
- [x] **Do not install `manifold-3d` yet**
  Hold it back until work actually begins on `Boolean3D` or union-style
  `MergeMeshes`.
- [x] **Create shared helpers before upgrading nodes**
  Add focused internal helpers for GLB loading/writing and metadata extraction
  so `FormatConverter`, `GetModel3DMetadata`, and later nodes do not each build
  their own parsing path.

Exit criteria for phase 0:

- `nodetool` builds with the new first-slice dependencies.
- There is one shared GLB/document helper path ready to reuse.

### 1. Make the existing I/O and metadata nodes honest

Start here. This is the first real implementation slice.

- [x] **Make `FormatConverter` a real converter**
  Keep GLB as the internal transport format. Replace byte-level or extension
  rename behavior with real parse-and-write flows for the formats we actually
  claim to support.
- [x] **Make `GetModel3DMetadata` read real geometry**
  Return actual bounds, primitive counts, mesh counts, material presence, and
  other cheap-to-compute facts from parsed model data rather than filename or
  container heuristics.
- [x] **Add fixture tests for these two nodes immediately**
  Do not defer tests to the end. Add at least one real GLB fixture and assert
  geometry-aware outputs.

Exit criteria for phase 1:

- `FormatConverter` no longer lies by only relabeling bytes.
- `GetModel3DMetadata` reports real geometry-derived values.
- Tests cover both nodes with real fixtures.

### 2. Upgrade simplification

- [x] **Make `Decimate` perform real simplification**
  Use `glTF-Transform` + `meshoptimizer` so triangle count changes are tied to
  mesh geometry, not placeholder behavior.
- [x] **Use the wrapper path first**
  Try `NodeIO` + `weld()` + `simplify()` before writing any custom low-level
  simplification plumbing.
- [x] **Keep v1 GLB-focused**
  Support real GLB decimation first. Do not block this phase on OBJ/STL/PLY.
- [x] **Document the truthful limits of `Decimate`**
  If only a subset of meshes or formats is supported at first, say so in node
  docs and metadata instead of pretending to support everything.
- [x] **Add fixture tests for `Decimate`**
  Assert that simplification changes geometry statistics in the expected
  direction, not just that some output bytes exist.

Exit criteria for phase 2:

- `Decimate` uses a real simplification path.
- Supported inputs and limitations are documented.
- Tests verify geometry-level effects.

### 3. Then add boolean / merge work

Only start this phase once phases 0-2 are stable.

- [x] **Install `manifold-3d`**
  Add it only when work on booleans actually starts.
- [x] **Make `Boolean3D` geometry-aware**
  Implement union / difference / intersection on real meshes or narrow the
  supported modes until the implementation is truthful.
- [x] **Make `MergeMeshes` explicit about behavior**
  Support at least one honest merge mode first: either scene merge
  (concatenate while preserving transforms) or boolean union. Do not keep one
  node that claims both if only one path is reliable.
- [x] **Add fixture tests for both nodes**
  Include success cases and at least one documented unsupported-input case.

Exit criteria for phase 3:

- `Boolean3D` and `MergeMeshes` have truthful behavior.
- Unsupported cases are explicit.
- Tests cover both success and failure/limit paths.

### 4. Refactor `model3d.ts` before adding more scope

This phase is complete.

- [x] **Split `model3d.ts` into a `model3d/` folder**
  Prefer a folder over more top-level flat files. Keep the public export surface
  stable through `src/nodes/model3d.ts`.
- [x] **Move low-level GLB helpers into a dedicated module**
  Put binary parsing/building and accessor utilities in something like
  `model3d/glb.ts`.
- [x] **Move document / conversion helpers into their own module**
  Put `glTF-Transform` I/O, conversion, merge, and decimation helpers in
  something like `model3d/document-ops.ts`.
- [x] **Move manifold boolean helpers into their own module**
  Put mesh extraction, Manifold bridge code, and boolean result rebuilding in
  something like `model3d/boolean-ops.ts`.
- [x] **Move shared types / small utilities out of the node file**
  Put `Model3DRefLike`, metadata types, format helpers, and byte helpers in
  `model3d/types.ts` and `model3d/utils.ts` or equivalent.
- [x] **Keep node classes together in a shallow node-facing module**
  Either keep all node classes in `model3d/nodes.ts` or split them by concern
  only if the split is obvious (`io-nodes.ts`, `mesh-nodes.ts`, `ai-nodes.ts`).
- [x] **Do not change node names or exports during the refactor**
  This should be structure cleanup, not another behavior change wave.
- [x] **Keep the current behavior suite green during the split**
  Use the existing focused GLB behavior tests as the guardrail for the refactor.

Exit criteria for phase 4:

- `model3d.ts` becomes a thin export/assembly layer or barrel.
- GLB/document/boolean code is separated from node class declarations.
- Existing focused `Model3D` behavior tests still pass after the split.

### 5. Add the missing cleanup nodes

This phase is complete.

- [x] **Keep node declarations separate from pure mesh helpers**
  `BaseNode` subclasses, `@prop` fields, node metadata, and registry exports stay
  in node-facing modules. Pure GLB/geometry logic should move into helper modules
  that can be tested without `BaseNode`.
- [x] **Do the mesh-edit helper split before adding more cleanup scope**
  Extract the long pure-geometry parts of `Transform3D`, `RecalculateNormals`,
  `CenterMesh`, and `FlipNormals` out of `model3d/nodes.ts` into a focused helper
  module such as `model3d/mesh-ops.ts`, while keeping public node exports stable.
- [x] **Do not add a deeper `model3d/nodes/` subfolder yet**
  Keep the folder flat until phase 5 adds enough new node families that a
  responsibility-based `nodes/` split is clearly justified.
- [x] **Add `NormalizeModel3D`**
  Provide one focused node for centering, orientation normalization, optional
  uniform scale-to-box, and optional ground-plane placement.
- Focused first pass landed in `nodetool` with explicit axis presets, optional
  scale-to-size, optional ground placement, and focused GLB behavior tests.
- [x] **Add component cleanup**
  Add `ExtractLargestComponent` and/or `RemoveSmallComponents` so workflows can
  remove floaters and disconnected junk geometry from AI-generated outputs.
- Focused first pass landed with `ExtractLargestComponent`, keeping the largest
  connected GLB triangle component by face count and covering the behavior with
  a disconnected-geometry fixture test.
- [x] **Add a conservative `RepairMesh`**
  Start with reliable fixes only: degenerate-face removal, duplicate or
  near-duplicate vertex merging, and similarly safe cleanup passes.
- Focused first pass landed with conservative GLB-only repair toggles for
  near-duplicate vertex welding plus degenerate-face removal.
- [x] **Add tests as each node lands**
  Do not batch all cleanup-node tests into a later PR.
- Focused fixture tests now cover both `ExtractLargestComponent` and
  `RepairMesh` behavior as each node landed.

Exit criteria for phase 5:

- Each new cleanup node has one narrow purpose.
- Each new cleanup node ships with focused fixture coverage.

### 6. Final verification pass

This is the next active phase.

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

