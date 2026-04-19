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

Work these items from top to bottom.

### 1. Upgrade the nodes that already exist

- [ ] **Make `FormatConverter` a real converter**
  Keep GLB as the internal transport format. Replace byte-level or extension
  rename behavior with real parse-and-write flows for the formats we actually
  claim to support.
- [ ] **Make `GetModel3DMetadata` read real geometry**
  Return actual bounds, primitive counts, mesh counts, material presence, and
  other cheap-to-compute facts from parsed model data rather than filename or
  container heuristics.
- [ ] **Make `Decimate` perform real simplification**
  Use a real simplification path so triangle count changes are tied to mesh
  geometry, not placeholder behavior.
- [ ] **Make `Boolean3D` geometry-aware**
  Implement union / difference / intersection on real meshes or narrow the
  supported modes until the implementation is truthful.
- [ ] **Make `MergeMeshes` explicit about behavior**
  Support at least one honest merge mode first: either scene merge
  (concatenate while preserving transforms) or boolean union. Do not keep one
  node that claims both if only one path is reliable.

### 2. Add the missing cleanup nodes needed by AI-generated meshes

- [ ] **Add `NormalizeModel3D`**
  Provide one focused node for centering, orientation normalization, optional
  uniform scale-to-box, and optional ground-plane placement.
- [ ] **Add component cleanup**
  Add `ExtractLargestComponent` and/or `RemoveSmallComponents` so workflows can
  remove floaters and disconnected junk geometry from AI-generated outputs.
- [ ] **Add a conservative `RepairMesh`**
  Start with reliable fixes only: degenerate-face removal, duplicate or
  near-duplicate vertex merging, and similarly safe cleanup passes.

### 3. Add verification before broadening scope

- [ ] **Add fixture-driven tests for every upgraded node**
  Include at least one real GLB fixture per node and assert geometry-level
  outcomes, not just non-empty bytes.
- [ ] **Document honest limits per node**
  If a node only supports triangular meshes, manifold inputs, or a subset of
  formats, state that in metadata and docs instead of implying universal mesh
  support.

---

## Explicitly later

- [ ] **UV unwrap / texture bake / retopology**
  Keep out of the first cleanup pass.
- [ ] **A broad "do everything" mesh pipeline**
  Prefer small trustworthy nodes over one oversized post-processing node.
- [ ] **MeshAnything V2 or similar post-processing models**
  Revisit only after the generic geometry node layer becomes reliable.

