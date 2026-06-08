# HuggingFace worker — Docker deployment

Run the HuggingFace node stack as a **remote Python worker** in a container, and
point a NodeTool TypeScript server at it. The TS server keeps owning the UI and API;
the heavy HuggingFace/ML work happens in the worker container.

This guide covers the **local CPU-only** path end to end (build → run → wire →
validate), the optional auth gate, and a summary of the cloud (RunPod / Vast.ai)
constraints for when you take this to a GPU host.

## How it fits together

```
web UI ──ws──► TS server (host :7777) ──► createPythonBridge()
                                              │  NODETOOL_WORKER_URL set?
                                              ▼  yes → WebsocketPythonBridge
                                          ws://localhost:8787  ──►  hf-worker container (:7777 internal)
                                          Authorization: Bearer <NODETOOL_WORKER_TOKEN>
```

- The worker is a long-lived **WebSocket server** (`python -m nodetool.worker --host
  0.0.0.0 --port 7777`), inherited from the `nodetool-core` image.
- The TS server attaches as a client whenever `NODETOOL_WORKER_URL` is set. No protocol
  change — the msgpack WebSocket bridge already exists.
- The container's internal port is `7777`, but the **TS dev server already owns host
  `7777`**, so the worker is published on host **`8787`** → `ws://localhost:8787`.

## Prerequisites

- **Docker** with Compose v2 (`docker compose`, not `docker-compose`).
- A **NodeTool TS server** checkout (the `nodetool2` repo) to act as the front end and
  bridge client.
- For the `nodetool-core:local` base image: the **`nodetool-core`** sibling repo
  checked out at `../nodetool-core` (relative to this repo), **or** a published image
  (`ghcr.io/nodetool-ai/nodetool:<tag>`) you re-tag as `nodetool-core:local`.
- Disk: the HF image is **multi-GB** (torch 2.9, diffusers, transformers, accelerate,
  bitsandbytes, …). The first build is slow; budget accordingly.
- **macOS note:** Docker on macOS is **CPU-only**. The smoke test below runs on CPU.
  GPU-only nodes (most diffusers / 3D) are expected to fail locally — that path is the
  cloud/GPU iteration, not this one.

## Build

The image layers HuggingFace on top of the core worker image:
`nodetool-core:local` (base worker) → `nodetool-hf:local` (adds the HF node stack).
The HF image installs the released `nodetool-huggingface` package from PyPI and inherits
`EXPOSE 7777`, the healthcheck, and `CMD` from core — it has no worker module of its own.

```bash
# 1. Build the base worker image (from the nodetool-core sibling repo).
make build-core
#   docker build -t nodetool-core:local ../nodetool-core
#
#   Alternatively, pull a published image and re-tag it:
#     docker pull ghcr.io/nodetool-ai/nodetool:<tag>
#     docker tag  ghcr.io/nodetool-ai/nodetool:<tag> nodetool-core:local

# 2. Build the HuggingFace worker image on top of it.
make build-hf
#   docker build -t nodetool-hf:local --build-arg CORE_IMAGE=nodetool-core:local .
```

`HF_VERSION` (default `0.7.1`) pins the PyPI release. To build a different version,
pass it through Docker directly:

```bash
docker build -t nodetool-hf:local \
  --build-arg CORE_IMAGE=nodetool-core:local \
  --build-arg HF_VERSION=0.7.1 .
```

Base HF dependencies only — optional extras (`ocr`, `hunyuan3d`, `triposg`, …) are not
installed. If you need them, add them as build-args or separate image variants.

## Run

```bash
make up      # docker compose up  — worker on localhost:8787 (foreground, logs to console)
make down    # docker compose down
```

`make up` starts the `hf-worker` service from `docker-compose.yaml`:

- Published on host **`8787`** → container `7777`.
- HuggingFace cache on a **named volume** `hf-cache` mounted at `/app/huggingface`
  (`HF_HOME=/app/huggingface`). Models downloaded on first run persist across restarts;
  treat it as a rebuildable warm cache.
- A WebSocket-handshake healthcheck (not an HTTP probe — the worker has no HTTP route),
  with a 60s `start_period` to allow warm-up. Wait for the container to report
  **healthy** before wiring the server:

  ```bash
  docker compose ps          # STATUS shows "healthy" once the WS handshake succeeds
  ```

The worker is published on `localhost` only by default. Do **not** expose port `8787`
beyond localhost without the auth gate below **and** TLS or a private tunnel (see
[Cloud deployment](#cloud-deployment-runpod--vastai)).

### GPU (not on macOS)

The default `hf-worker` service is CPU. A separate `hf-worker-gpu` service (compose
profile `gpu`) reserves an NVIDIA device and requires the host NVIDIA driver plus
container runtime:

```bash
docker compose --profile gpu up hf-worker-gpu
```

This is not usable on macOS Docker.

## Wire the TS server to the worker

Setting `NODETOOL_WORKER_URL` switches the whole TS server onto the remote worker:
`createPythonBridge()` returns a reconnecting `WebsocketPythonBridge` instead of
spawning a local stdio worker. Start the server (in the `nodetool2` repo) with:

```bash
NODETOOL_WORKER_URL=ws://localhost:8787 \
NODETOOL_WORKER_TOKEN=<secret> \
npm run dev:server
```

- `NODETOOL_WORKER_URL=ws://localhost:8787` — the published worker.
- `NODETOOL_WORKER_TOKEN=<secret>` — only needed if the worker was started with a token
  (see below). The bridge sends it as `Authorization: Bearer <secret>` on the handshake
  (initial connect **and** every reconnect). Leave it unset for open local/dev.

On a successful attach the server logs a `discover` + `worker.status`, and HuggingFace
node metadata becomes available in the editor.

### Authentication (`NODETOOL_WORKER_TOKEN`)

A shared-secret bearer token, **opt-in**, same env var name on both ends:

- **Unset / empty** → the worker accepts all connections and the client sends no header
  (preserves local/stdio/dev behavior). This is the default.
- **Set** → the worker requires `Authorization: Bearer <token>` on the WS handshake and
  compares it in constant time; a missing/wrong token is rejected with **HTTP 401**
  before any application frame. The client must send the matching token.

Set the token on the worker via the environment the compose file passes through:

```bash
NODETOOL_WORKER_TOKEN=<secret> make up
# or export it once: export NODETOOL_WORKER_TOKEN=<secret>; make up
```

Then start the server with the **same** value (see the wire command above).

Why a header (not a first message or a subprotocol): both libraries support it
directly, the handshake is rejected before the executor is ever reached, and headers
survive direct-TCP and HTTP-proxy paths alike.

## Validate (local, CPU-only)

The smoke workflow exercises a real HuggingFace node on the remote worker:

- DSL form (documents the graph): [`examples/hf-worker-smoke.ts`](../examples/hf-worker-smoke.ts)
- JSON artifact (loaded into the server): [`examples/hf-worker-smoke.json`](../examples/hf-worker-smoke.json)

It runs two `SentenceSimilarity` nodes
(`sentence-transformers/all-MiniLM-L6-v2`, ~80 MB, CPU-fast) — one per sentence — and
previews the resulting np_array embeddings. First run downloads the model into the
`hf-cache` volume; subsequent runs reuse it.

> **Run it through a running server — not `nodetool run`.** The remote
> `WebsocketPythonBridge` is created **only** by the websocket server. The CLI/DSL local
> path (`nodetool run <file>`) does **not** create a Python bridge, so it would never
> reach the worker. Load the workflow into a server that has `NODETOOL_WORKER_URL` set
> (per the spec, §5.5).

Steps:

1. **Build** — `make build-core`, then `make build-hf`.
2. **Run** — `NODETOOL_WORKER_TOKEN=<secret> make up`; wait for `docker compose ps` to
   report **healthy**.
3. **Auth gate** — with the worker started with a token, confirm the gate:
   - Server with **no** or **wrong** `NODETOOL_WORKER_TOKEN` → handshake rejected (401),
     the bridge fails to attach (worker logs the rejected handshake).
   - Server with the **matching** token → attaches; `discover` + `worker.status` logged.
4. **Wire** — start the TS server with `NODETOOL_WORKER_URL=ws://localhost:8787` and the
   matching `NODETOOL_WORKER_TOKEN` (see wire command above). Confirm HF node metadata is
   present.
5. **Run the graph through the server** — load `examples/hf-worker-smoke.json` into the
   running server and execute it:
   - **Web UI:** import the JSON in the editor and run it; the two Preview nodes show the
     embeddings.
   - **Server run API:** POST the graph to the server's run endpoint.

   The `SentenceSimilarity` nodes execute **on the container**, and the embeddings come
   back through the bridge. That round trip proves: image health, the auth gate, bridge
   attach, and a real HF graph running on the remote worker.

CUDA-only nodes (most diffusers / 3D) are expected to fail on macOS Docker and are
deferred to the GPU iteration.

## Caveats

- **CPU-only on macOS.** Docker on macOS has no GPU passthrough. Validate only
  CPU-capable nodes locally; GPU validation belongs to the cloud iteration.
- **Host port `8787`, not `7777`.** The TS dev server binds host `7777`, so the worker is
  published on `8787` → `NODETOOL_WORKER_URL=ws://localhost:8787`.
- **First-run model download latency.** The first run of a model pays the download cost;
  the `hf-cache` named volume persists it across restarts. The healthcheck's
  `start_period` covers warm-up.
- **Token unset = open.** With no `NODETOOL_WORKER_TOKEN`, the worker accepts every
  connection. That is fine on localhost; it is **remote code execution** if the port is
  reachable from elsewhere. Never expose the worker without the token **and** transport
  protection (next section).
- **Large image / slow first build.** Expected, given the ML stack. Layering on
  `nodetool-core` maximizes cache reuse between rebuilds.

## Cloud deployment (RunPod / Vast.ai)

The **image** design holds unchanged on GPU hosts (PyPI CUDA wheels bundle their own CUDA
runtime; the host supplies only the NVIDIA driver; `CMD` runs on boot). The deltas below
are exposure / ops concerns. These are **recorded constraints for the next iteration** —
dynamic endpoint resolution and a first-class deploy target are not built here.

- **Transport: direct TCP, never an HTTP proxy.**
  - **RunPod** — its HTTP proxy (`*.proxy.runpod.net`) is Cloudflare-fronted (100 MB body
    cap, 100 s timeout, WS drops at ~1.8 MiB) and unusable for large binary frames. Use
    **Expose TCP Ports** (direct, public IP, random external port).
  - **Vast.ai** — raw `-p` is direct TCP passthrough (good). Its Caddy auth-proxy (used
    when external ≠ internal port) inserts an HTTP hop; avoid it with an **identity port
    map** (e.g. a port > 70000) for raw passthrough.
  - Prefer offloading large blobs to S3 / `ASSET_BUCKET` so WebSocket frames stay small;
    treat the 256 MB bridge frame ceiling as a backstop, not a routine payload size.

- **Dynamic endpoint.** Both platforms map internal `7777` to a **random external port on
  a public IP, known only after boot** (RunPod: `RUNPOD_PUBLIC_IP` / `RUNPOD_TCP_PORT_*`;
  Vast: "IP Port Info" / API). `NODETOOL_WORKER_URL` must be resolved **at runtime from
  the provider API**, not hardcoded.
  - **Known gap:** the current bridge reconnects to a **fixed** URL. For cloud, reconnect
    must re-resolve host:port (instance/IP can change). Deploy-iteration work.

- **Never expose the raw port.** A world-reachable port + no app auth = public RCE on the
  GPU. The `NODETOOL_WORKER_TOKEN` gate is the app-level control, but for public exposure
  it must be paired with **TLS (`wss`)** terminated by you, or — preferably — a **private
  tunnel** (Tailscale / WireGuard / SSH) with the worker bound to the tunnel interface.
  Never expose a token-only port over plain `ws://` on the public internet.

- **GPU / host selection.** Filter for **CUDA ≥ 12.x** (torch 2.9) — RunPod
  `allowedCudaVersions` / UI filter; Vast `cuda_vers>=… driver_version>=…` — or the
  container can land on an old-driver host and fail to use the GPU.

- **Launch mode.** RunPod = **Pod**, not Serverless (serverless can't hold a persistent
  client WebSocket). Vast = **entrypoint** launch mode (so the image `CMD` runs as PID 1;
  SSH/Jupyter modes replace it — use `onstart` there).

- **Model cache.** RunPod Network Volume mounts at `/workspace`
  (`HF_HOME=/workspace/.cache/huggingface`); Vast Volume mounts at `/data` (host-pinned,
  lost if the host vanishes); Vast container disk is wiped on destroy. Treat the HF cache
  as a rebuildable warm cache, not durable storage.

- **Cost / reliability.** A persistent GPU bills continuously (no scale-to-zero). Vast
  hosts are heterogeneous and interruptible — prefer on-demand, high-reliability hosts and
  treat the worker as ephemeral.
