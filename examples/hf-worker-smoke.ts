/**
 * HuggingFace worker smoke test (DSL form)
 *
 * Purpose: the smallest possible graph that exercises a real HuggingFace node
 * on a *remote* worker container, for the local CPU-only validation in
 * docs/worker-deployment.md (spec 2026-06-08-hf-worker-docker-design.md, §5.5/§7).
 *
 * This `.ts` file *documents* the graph. The artifact actually loaded into the
 * running TS server is the sibling `hf-worker-smoke.json` (UI import or the
 * server's run API). The remote `WebsocketPythonBridge` is created only by the
 * websocket server, so the graph must run *through a server* with
 * `NODETOOL_WORKER_URL=ws://localhost:8787` set — not via `nodetool run`.
 *
 * Graph:
 *   SentenceSimilarity("The cat sits on the mat.")  ──► Preview (embedding A)
 *   SentenceSimilarity("A feline rests on the rug.") ──► Preview (embedding B)
 *
 * Model: sentence-transformers/all-MiniLM-L6-v2 (~80 MB, CPU-fast). First run
 * downloads it into the hf-cache volume; subsequent runs reuse it.
 *
 * Node-shape note: the HuggingFace `SentenceSimilarity` node
 * (huggingface.sentence_similarity.SentenceSimilarity) takes a single `inputs`
 * string and returns a single `np_array` *embedding* — it does NOT compute a
 * similarity score between two strings. There is no verified cosine-similarity
 * node in the ecosystem to fold the two embeddings into a scalar, so this smoke
 * test emits the two embeddings (one per string) instead of a single score.
 * The two-string intent is preserved; the "output similarity score" reduction
 * is a follow-up (see deviations in the implementation record).
 *
 * The HF node has no typed DSL factory (the HuggingFace pack is not part of the
 * generated DSL surface), so it is built with the generic `createNode` helper
 * using its fully-qualified node type. `Preview` does have a typed factory.
 *
 * Tags: huggingface, smoke, embeddings, example
 */

import {
  workflow,
  createNode,
  workflowsBase_node,
  type SingleOutput,
} from "@nodetool-ai/dsl";

// The HF node emits an np_array embedding `{ type: "np_array", value, dtype, shape }`.
// The DSL does not re-export protocol's NPArray, so carry it as a plain record.
type NPArray = Record<string, unknown>;

const MODEL = {
  type: "hf.sentence_similarity",
  repo_id: "sentence-transformers/all-MiniLM-L6-v2",
};

// huggingface.sentence_similarity.SentenceSimilarity — no typed factory, so
// build it generically. `inputs` is the text; the single output slot is
// `output`, carrying an np_array embedding.
function sentenceSimilarity(inputs: { model: unknown; inputs: string }) {
  return createNode<SingleOutput<NPArray>>(
    "huggingface.sentence_similarity.SentenceSimilarity",
    inputs as Record<string, unknown>,
  );
}

const embeddingA = sentenceSimilarity({
  model: MODEL,
  inputs: "The cat sits on the mat.",
});

const embeddingB = sentenceSimilarity({
  model: MODEL,
  inputs: "A feline rests on the rug.",
});

const previewA = workflowsBase_node.preview({
  value: embeddingA.output(),
  name: "embedding_a",
});

const previewB = workflowsBase_node.preview({
  value: embeddingB.output(),
  name: "embedding_b",
});

const wf = workflow(previewA, previewB);
console.log(JSON.stringify(wf, null, 2));
