"""Regenerate `recommended_models` in the package metadata from the node sources.

`nodetool package scan` needs nodetool-core (and torch/diffusers) importable, which
is not available here, so this evaluates the `get_recommended_models` classmethod
bodies with a small AST interpreter instead. It handles exactly the shapes used in
this repo: literal constructor calls and list comprehensions over literal lists.
"""

import ast
import json
import pathlib
import sys

NODES = pathlib.Path("src/nodetool/nodes/huggingface")
META = pathlib.Path("src/nodetool/package_metadata/nodetool-huggingface.json")


class Unsupported(Exception):
    pass


def literal(node, env):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [literal(e, env) for e in node.elts]
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.Attribute):  # e.g. SomeEnum.MEMBER
        raise Unsupported("attribute")
    raise Unsupported(ast.dump(node)[:60])


def call_to_entry(node, env):
    """Turn `HFSomething(repo_id=..., allow_patterns=[...])` into a metadata dict."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise Unsupported("call")
    entry = {"__cls__": node.func.id}
    for kw in node.keywords:
        if kw.arg is None:
            raise Unsupported("**kwargs")
        entry[kw.arg] = literal(kw.value, env)
    if "repo_id" not in entry:
        raise Unsupported("no repo_id")
    return entry


def eval_return(node, env):
    """Evaluate the list expression of a `return [...]`."""
    out = []
    if isinstance(node, ast.ListComp):
        return eval_listcomp(node, env)
    if isinstance(node, ast.Name):
        entries = env.get("__lists__", {}).get(node.id)
        if entries is None:
            raise Unsupported(f"unknown name {node.id}")
        return [call_to_entry(e, env) for e in entries]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return eval_return(node.left, env) + eval_return(node.right, env)
    # `super().get_recommended_models()` — walk up to the base class definition.
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_recommended_models"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
    ):
        resolve = env.get("__super__")
        if resolve is None:
            raise Unsupported("super() with no base")
        return resolve()
    if not isinstance(node, ast.List):
        raise Unsupported("return not a list")
    for elt in node.elts:
        if isinstance(elt, ast.Starred):
            out.extend(eval_return(elt.value, env))
            continue
        if isinstance(elt, ast.Call):
            out.append(call_to_entry(elt, env))
        elif isinstance(elt, ast.ListComp):
            out.extend(eval_listcomp(elt, env))
        else:
            raise Unsupported("elt")
    return out


def eval_listcomp(node, env):
    if len(node.generators) != 1:
        raise Unsupported("multi-generator comp")
    gen = node.generators[0]
    if gen.ifs or not isinstance(gen.target, ast.Name):
        raise Unsupported("comp with ifs")
    items = literal(gen.iter, env)
    return [call_to_entry(node.elt, {**env, gen.target.id: item}) for item in items]


def collect_source_models():
    """node class name -> list of entries, per module."""
    result = {}
    # Model lists are shared across modules by import (e.g. HF_CONTROLNET_MODELS
    # lives in stable_diffusion_base but is used from image_to_image), so gather
    # them from every module up front.
    shared_lists = {}
    classes = {}
    owners = {}
    bases = {}
    for path in sorted(NODES.glob("*.py")):
        for stmt in ast.parse(path.read_text()).body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target, value = stmt.targets[0], stmt.value
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                target, value = stmt.target, stmt.value
            else:
                continue
            if (
                isinstance(target, ast.Name)
                and isinstance(value, ast.List)
                and value.elts
                and all(isinstance(e, ast.Call) for e in value.elts)
            ):
                shared_lists[target.id] = value.elts

    for path in sorted(NODES.glob("*.py")):
        tree = ast.parse(path.read_text())
        module_consts = {"__lists__": shared_lists}
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                # Module-level list of model constructor calls, e.g.
                # HF_STABLE_DIFFUSION_MODELS = [HFStableDiffusion(...), ...]
                if target.id in shared_lists:
                    continue
                try:
                    module_consts[target.id] = literal(stmt.value, {})
                except Unsupported:
                    pass
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            bases[cls.name] = [b.id for b in cls.bases if isinstance(b, ast.Name)]
            owners.setdefault(cls.name, path.stem)
            for fn in cls.body:
                if (
                    not isinstance(fn, ast.FunctionDef)
                    or fn.name != "get_recommended_models"
                ):
                    continue
                returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
                if len(returns) != 1 or returns[0].value is None:
                    continue
                classes[cls.name] = (returns[0].value, fn, module_consts, cls)
                owners[cls.name] = path.stem

    def entries_for(class_name, seen=()):
        if class_name in seen:
            raise Unsupported(f"cycle at {class_name}")
        if class_name not in classes:
            # Class inherits get_recommended_models — walk up to the definer.
            for base in bases.get(class_name, []):
                try:
                    return entries_for(base, seen + (class_name,))
                except Unsupported:
                    continue
            raise Unsupported(f"unresolvable base {class_name}")
        ret, fn, module_consts, cls = classes[class_name]
        env = dict(module_consts)
        for stmt in fn.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    try:
                        env[target.id] = literal(stmt.value, env)
                    except Unsupported:
                        pass
        parents = bases.get(class_name) or []
        if parents:
            env["__super__"] = lambda: entries_for(parents[0], seen + (class_name,))
        return eval_return(ret, env)

    for class_name, module in owners.items():
        try:
            result[f"huggingface.{module}.{class_name}"] = entries_for(class_name)
        except Unsupported as exc:
            # Only interesting for classes that do declare the method themselves;
            # everything else simply has no recommended models.
            if class_name in classes:
                print(f"  skip {module}.{class_name}: {exc}")
    return result


def main():
    meta = json.loads(META.read_text())
    source = collect_source_models()

    # Learn the metadata "type" string for each model python class (HFControlNet
    # -> hf.controlnet, ...) from the entries the generator already produced.
    cls_types = {}
    for node in meta["nodes"]:
        entries = source.get(node["node_type"]) or source.get(
            node["node_type"] + "Node"
        )
        if not entries:
            continue
        by_repo = {m["repo_id"]: m for m in (node.get("recommended_models") or [])}
        for entry in entries:
            old = by_repo.get(entry["repo_id"])
            if old and old.get("type"):
                cls_types.setdefault(entry["__cls__"], old["type"])

    changed = 0
    for node in meta["nodes"]:
        # A few node classes are exported under a name without the "Node" suffix.
        entries = source.get(node["node_type"]) or source.get(
            node["node_type"] + "Node"
        )
        if entries is None:
            continue
        existing = node.get("recommended_models") or []
        # Map the python class of a recommended model to the metadata "type"
        # string by reusing whatever the already-generated metadata used.
        by_repo = {m["repo_id"]: m for m in existing}
        types = {m.get("type") for m in existing}
        default_type = types.pop() if len(types) == 1 else None
        new = []
        for entry in entries:
            cls = entry.pop("__cls__")
            old = by_repo.get(entry["repo_id"])
            model_type = (old or {}).get("type") or cls_types.get(cls) or default_type
            if model_type is None:
                print(f"  ! no type for {node['node_type']} {entry['repo_id']} ({cls})")
                return 1
            out = {
                "id": entry["repo_id"],
                "name": entry["repo_id"],
                "repo_id": entry["repo_id"],
                "type": model_type,
            }
            for key in ("allow_patterns", "ignore_patterns", "path"):
                if key in entry:
                    out[key] = entry[key]
                elif old is not None and key in old:
                    out[key] = old[key]
            if "path" in out:
                out["id"] = f"{out['repo_id']}:{out['path']}"
            new.append(dict(sorted(out.items())))
        if new != existing:
            node["recommended_models"] = new
            changed += 1
    META.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"updated {changed} node entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
