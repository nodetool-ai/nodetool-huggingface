import asyncio
from enum import Enum
import re
import logging
from nodetool.metadata.types import Provider, ImageRef, AudioRef, VideoRef, NPArray
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from typing import Any, Type, get_origin
import torch
from nodetool.workflows.types import NodeProgress, NodeUpdate, LogUpdate

from nodetool.nodes.huggingface.prediction import run_huggingface

try:
    from transformers.utils import logging as hf_logging
    from diffusers.utils import logging as diffusers_logging

    HF_LOGGING_AVAILABLE = True
except ImportError:
    HF_LOGGING_AVAILABLE = False


class HuggingFaceLogHandler(logging.Handler):
    """Custom logging handler that redirects HuggingFace logs to LogUpdate events."""

    def __init__(self, context: ProcessingContext, node_id: str, node_name: str):
        super().__init__()
        self.context = context
        self.node_id = node_id
        self.node_name = node_name

    def emit(self, record: logging.LogRecord):
        """Emit a log record as a LogUpdate event."""
        try:
            message = self.format(record)
            # Map log levels to severity
            severity_map = {
                "DEBUG": "info",
                "INFO": "info",
                "WARNING": "warning",
                "ERROR": "error",
                "CRITICAL": "error",
            }
            severity = severity_map.get(record.levelname, "info")

            log_update = LogUpdate(
                node_id=self.node_id,
                node_name=self.node_name,
                content=message,
                severity=severity,  # type: ignore
            )
            self.context.post_message(log_update)
        except Exception as e:
            # Avoid infinite recursion if there's an error in logging
            pass


def setup_hf_logging(
    context: ProcessingContext, node_id: str, node_name: str = ""
) -> HuggingFaceLogHandler | None:
    """Set up HuggingFace logging to redirect to LogUpdate events."""
    if not HF_LOGGING_AVAILABLE:
        return None

    # Create custom handler
    handler = HuggingFaceLogHandler(context, node_id, node_name)
    handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))

    # Add handler to transformers logger
    transformers_logger = hf_logging.get_logger()
    transformers_logger.addHandler(handler)
    transformers_logger.setLevel(logging.INFO)

    # Add handler to diffusers logger
    diffusers_logger = diffusers_logging.get_logger()
    diffusers_logger.addHandler(handler)
    diffusers_logger.setLevel(logging.INFO)

    # Add handler to various HuggingFace related loggers
    logger_names = [
        "transformers",
        "diffusers",
        "huggingface_hub",
        "accelerate",
        "safetensors",
        "tokenizers",
    ]

    for logger_name in logger_names:
        try:
            logger = logging.getLogger(logger_name)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            # Ensure logs propagate to our handler
            logger.propagate = True
        except Exception:
            pass

    # Set appropriate log levels
    hf_logging.set_verbosity_info()
    diffusers_logging.set_verbosity_info()

    # Also try to force enable some detailed logging
    try:
        logging.getLogger("huggingface_hub.file_download").setLevel(logging.INFO)
        logging.getLogger("huggingface_hub.snapshot_download").setLevel(logging.INFO)
    except Exception:
        pass

    return handler


def progress_callback(node_id: str, total_steps: int, context: ProcessingContext):
    def callback(step: int, timestep: int, latents: torch.FloatTensor) -> None:
        context.post_message(
            NodeProgress(
                node_id=node_id,
                progress=step,
                total=total_steps,
            )
        )

    return callback
