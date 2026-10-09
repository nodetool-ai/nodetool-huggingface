# HuggingFace worker — Docker build/run wrappers.
#
# Image layering: nodetool-core:local (base worker) -> nodetool-hf:local (adds
# the HuggingFace node stack). build-hf depends on build-core having produced
# nodetool-core:local. ghcr.io/nodetool-ai/nodetool:<tag> is the TypeScript
# server image, not a Python worker, so it cannot stand in for it.
#
# Local run publishes the worker on host port 8787 (the TS server owns 7777):
#   NODETOOL_WORKER_URL=ws://localhost:8787
# Set NODETOOL_WORKER_TOKEN to require "Authorization: Bearer <token>" on the
# handshake; leave it unset for open local/dev access.
.PHONY: build-core build-hf up down

build-core:
	docker build -t nodetool-core:local ../nodetool-core

build-hf:
	docker build -t nodetool-hf:local --build-arg CORE_IMAGE=nodetool-core:local .

up:
	docker compose up

down:
	docker compose down
