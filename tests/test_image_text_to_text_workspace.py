"""Regression: ImageTextToText failed with "No workspace is assigned" on a
worker that DID have one.

On a rented worker (RunPod), `nodetool.worker.executor.run_node` assigns a
temp-dir workspace to the `ProcessingContext` before running the node, and
input blobs are written into that workspace and handed to the node as
`file://` URIs. `Qwen3_VL`, in the same module, reads its image with
`context.image_to_pil(self.image)` and resolves such a URI fine, because
`image_to_pil` -> `asset_to_io` -> `download_file` resolves `file://` paths
through `context.resolve_workspace_path`, which uses `context.workspace_dir`.

`ImageTextToText` instead built a `MessageImageContent` and handed it to
`HuggingFaceLocalProvider.generate_messages`. That provider's
`convert_message` resolved the image via a bare
`fetch_uri_bytes_and_mime_async(uri)` call with no `workspace_dir` argument,
so the module-level default (`None`) was used regardless of whether the
context actually had a workspace -- and a `file://` URI raised "No workspace
is assigned" even when one was assigned. Because this only fires once
`convert_message` runs, the model had already finished downloading (tens of
minutes, tens of gigabytes) by the time it did.

The fix threads `context` through `convert_message` so it resolves the image
via `context.image_to_pil`, the same workspace-aware route `Qwen3_VL` already
uses, and adds `ImageTextToText.pre_process` so a genuinely bad image
reference fails before `preload_model` downloads anything.
"""

import sys
from io import BytesIO
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import PIL.Image
import pytest

from nodetool.huggingface.huggingface_local_provider import HuggingFaceLocalProvider
from nodetool.metadata.types import ImageRef, Message, MessageImageContent, MessageTextContent
from nodetool.nodes.huggingface.image_text_to_text import ImageTextToText
from nodetool.workflows.processing_context import ProcessingContext


def _write_worker_style_image(workspace_dir: str) -> ImageRef:
    """Mirror what `nodetool.worker.executor._prepare_node` does with an
    input blob: write it into the assigned workspace and hand the node a
    workspace-relative `file:///<filename>` URI -- not an absolute path --
    the same shape `input_ref_uris` builds for every media input."""
    path = Path(workspace_dir) / "input_image.png"
    buffer = BytesIO()
    PIL.Image.new("RGB", (4, 4), color="red").save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())
    return ImageRef(uri="file:///input_image.png")


@pytest.mark.asyncio
async def test_convert_message_resolves_file_uri_against_assigned_workspace(tmp_path):
    """The core bug: a file:// image in an assigned workspace must resolve,
    not raise "No workspace is assigned"."""
    context = ProcessingContext(workspace_dir=str(tmp_path))
    image = _write_worker_style_image(str(tmp_path))
    message = Message(
        role="user",
        content=[
            MessageImageContent(image=image),
            MessageTextContent(text="Describe this image."),
        ],
    )

    provider = HuggingFaceLocalProvider()
    converted = await provider.convert_message(message, context=context)
    # Note on what this proves. Against the unfixed code this test fails with
    # `TypeError: unexpected keyword argument 'context'` -- which shows the new
    # parameter is missing, not that the reported bug occurred. The bug itself
    # is pinned by test_the_old_context_free_route_still_raises below, which
    # calls the pre-fix route and asserts the exact failure.

    assert converted["role"] == "user"
    image_parts = [part for part in converted["content"] if part.get("type") == "image"]
    assert len(image_parts) == 1
    assert isinstance(image_parts[0]["image"], PIL.Image.Image)


@pytest.mark.asyncio
async def test_pre_process_fails_before_model_download_on_bad_image(tmp_path, monkeypatch):
    """`pre_process` must surface an image-resolution failure itself, before
    `preload_model` (and its multi-gigabyte download) ever runs."""
    context = ProcessingContext(workspace_dir=str(tmp_path))
    node = ImageTextToText()
    # No data and a file:// URI that does not exist inside the workspace.
    node.image = ImageRef(uri="file:///missing.png")

    preload_calls: list[bool] = []

    async def _fake_preload_model(self, context):  # pragma: no cover - must not run
        preload_calls.append(True)

    monkeypatch.setattr(ImageTextToText, "preload_model", _fake_preload_model)

    with pytest.raises(FileNotFoundError):
        await node.pre_process(context)

    assert preload_calls == []


@pytest.mark.asyncio
async def test_the_old_context_free_route_still_raises(tmp_path):
    """Pin the reported failure, not just the absence of the new parameter.

    Dropping the context is what the pre-fix `convert_message` did: it called
    `fetch_uri_bytes_and_mime_async(uri)` with no workspace_dir. Reproduced on
    unfixed main, with a workspace assigned:

        PermissionError: No workspace is assigned. File operations require a
        user-defined workspace.

    Calling the same route with context=None must still raise, so a later
    refactor that quietly reintroduces the context-free path is caught here
    rather than after a multi-gigabyte download on a worker.
    """
    context = ProcessingContext(workspace_dir=str(tmp_path))
    image = _write_worker_style_image(str(tmp_path))
    message = Message(
        role="user",
        content=[
            MessageImageContent(image=image),
            MessageTextContent(text="Describe this image."),
        ],
    )

    provider = HuggingFaceLocalProvider()

    with pytest.raises((PermissionError, FileNotFoundError, ValueError)):
        await provider.convert_message(message, context=None)
