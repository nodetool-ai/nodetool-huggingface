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
                'DEBUG': 'info',
                'INFO': 'info', 
                'WARNING': 'warning',
                'ERROR': 'error',
                'CRITICAL': 'error'
            }
            severity = severity_map.get(record.levelname, 'info')
            
            log_update = LogUpdate(
                node_id=self.node_id,
                node_name=self.node_name,
                content=message,
                severity=severity
            )
            self.context.post_message(log_update)
        except Exception as e:
            # Avoid infinite recursion if there's an error in logging
            pass


def setup_hf_logging(context: ProcessingContext, node_id: str, node_name: str = "") -> HuggingFaceLogHandler | None:
    """Set up HuggingFace logging to redirect to LogUpdate events."""
    if not HF_LOGGING_AVAILABLE:
        return None
        
    # Create custom handler
    handler = HuggingFaceLogHandler(context, node_id, node_name)
    handler.setFormatter(logging.Formatter('%(name)s - %(message)s'))
    
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
        "tokenizers"
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


def convert_enum_value(value: Any):
    """
    Converts an enum value to its corresponding value.

    Args:
        value (Any): The value to be converted.

    Returns:
        Any: The converted value.
    """
    if isinstance(value, Enum):
        return value.value
    return value


def sanitize_enum(name: str) -> str:
    """
    Sanitizes an enum string by replacing hyphens, dots, and spaces with underscores.

    Args:
        enum (str): The enum string to be sanitized.

    Returns:
        str: The sanitized enum string.
    """
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name).upper()
    while len(name) > 0 and name[0] == "_":
        name = name[1:]
    while len(name) > 0 and name[-1] == "_":
        name = name[:-1]
    if name[0].isdigit():
        name = "_" + name
    return name


async def convert_output_value(
    value: Any, t: Type[Any], output_index: int = 0, output_key: str = "output"
):
    """
    Converts the output value to the specified type.
    Performs automatic conversions using heuristics.

    Args:
        value (Any): The value to be converted.
        t (Type[Any]): The target type to convert the value to.
        output_index: The index for list outputs to use.

    Returns:
        Any: The converted value.

    Raises:
        TypeError: If the value is not of the expected type.

    """
    if value is None:
        return None

    if isinstance(value, t):
        return value

    if t in (ImageRef, AudioRef, VideoRef):
        if type(value) == list:
            return t(uri=str(value[output_index])).model_dump()
        elif type(value) == dict:
            if output_key in value:
                return t(uri=value[output_key]).model_dump()
            else:
                if len(value) == 0:
                    raise ValueError(f"Invalid {t} value: {value}")
                uri = list(value.values())[0]
                return t(uri=uri).model_dump()
        elif type(value) == str:
            return t(uri=value).model_dump()
        else:
            raise TypeError(f"Invalid {t} value: {value}")
    elif t == str:
        if type(value) == list:
            return "".join(str(i) for i in value)
        else:
            return str(value)
    elif t == NPArray:
        return NPArray.from_list(value).model_dump()
    elif get_origin(t) == list:
        if type(value) != list:
            raise TypeError(f"value is not list: {value}")
        tasks = [convert_output_value(item, t.__args__[0]) for item in value]  # type: ignore
        return await asyncio.gather(*tasks)
    return value


class HuggingfaceNode(BaseNode):

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not HuggingfaceNode

    async def pre_process(self, context: ProcessingContext):
        """Base pre_process that sets up HuggingFace logging for all HF nodes."""
        # Set up HuggingFace logging redirection automatically
        setup_hf_logging(context, self.id, self.get_title())
        # Call parent implementation
        await super().pre_process(context)

    async def run_huggingface(
        self,
        model_id: str,
        context: ProcessingContext,
        params: dict[(str, Any)] | None = None,
        data: bytes | None = None,
    ) -> Any:
        if params and data:
            raise ValueError("Cannot provide both params and data to run_huggingface.")
        raw_inputs = {
            prop.name: convert_enum_value(getattr(self, prop.name))
            for prop in self.properties()
        }
        raw_inputs = {**raw_inputs, **(params or {})}

        input_params = {
            prop.name: await context.convert_value_for_prediction(
                prop,
                getattr(self, prop.name),
            )
            for prop in self.properties()
        }
        input_params = {
            key: value for (key, value) in input_params.items() if (value is not None)
        }
        input_params = {**input_params, **(params or {})}

        context.post_message(
            NodeUpdate(
                node_id=self.id,
                node_name=model_id,
                node_type=self.get_node_type(),
                status="starting",
            )
        )

        return await context.run_prediction(
            provider=Provider.HuggingFace,
            node_id=self.id,
            model=model_id,
            params=input_params,
            run_prediction_function=run_huggingface,
            data=data,
        )

    async def extra_params(self, context: ProcessingContext) -> dict:
        return {}

    async def convert_output(self, context: ProcessingContext, output: Any) -> Any:
        t = self.return_type()
        if isinstance(t, dict):
            return output
        if isinstance(t, Type):
            output = await convert_output_value(output, t)
        return {
            "output": output,
        }


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
