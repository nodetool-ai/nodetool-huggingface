# Model3D core future

This file tracks future work that would belong in `nodetool-core` or
`nodetool-sdk`.

There is no required core or SDK work for the active local 3D generator fix
plan. Keep this file as a parking lot for changes that become relevant only
after we choose to support richer 3D asset shapes.

---

## Current status

- No blocking `nodetool-core` tasks for `FIX-MODEL3D.md`.
- No current `nodetool-sdk` implementation work.
- Re-open this file only when a concrete generator or node requires a new
  shared type, serialization contract, or discovery rule.

---

## Future candidates

### 1. Multi-mesh outputs for part-based generators

Trigger: adopt something like `PartCrafter` or any model that returns several
editable meshes as one logical result.

- [ ] **Decide whether `Model3DRef` should stay single-asset**
  Confirm whether multi-part output belongs as a new ref type rather than
  overloading the existing single-model contract.
- [ ] **If needed, add a part-list type**
  Likely shape: `Model3DPartListRef` or another explicit collection type with
  stable per-part names, optional labels, transforms, and metadata.
- [ ] **Define serialization and UI expectations**
  If a part-list type lands, specify how it is stored, previewed, iterated, and
  passed between nodes.

### 2. Non-mesh 3D asset families

Trigger: decide to support outputs such as 3D Gaussian splats as first-class
workflow assets instead of flattening everything into meshes.

- [ ] **Decide whether a new ref type is warranted**
  Do not overload `Model3DRef` with fundamentally different asset semantics.
- [ ] **Define minimum viewer and metadata support before adoption**
  A new asset type is not useful unless preview, persistence, and export
  expectations are clear.

### 3. Shared metadata contract upgrades

Trigger: two or more repos need the same additional 3D metadata fields.

- [ ] **Promote only stable fields into shared types**
  Examples could include canonical orientation, unit scale, mesh count, or
  resolved seed provenance, but only after those fields are proven useful in
  real workflows.
- [ ] **Avoid speculative schema expansion**
  Do not add shared fields just because one experimental node might want them.

---

## Explicitly not now

- Changing `nodetool-sdk` discovery only for the current HF split.
- Adding new shared 3D abstractions before there is a concrete second consumer.
- Expanding core types just to mirror model-specific research notes.

