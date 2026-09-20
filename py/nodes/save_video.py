from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import folder_paths  # pyright: ignore[reportMissingImports]

from ..metadata_collector import get_metadata
from ..metadata_collector.metadata_processor import MetadataProcessor
from .save_image import SaveImageLM

try:  # ComfyUI builds with the VIDEO type expose the container/codec enums here
    from comfy_api.latest import Types as _ComfyTypes  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - older ComfyUI or standalone/test environments
    _ComfyTypes = None

logger = logging.getLogger(__name__)

VIDEO_FORMATS = ["auto", "mp4", "mkv", "webm"]
VIDEO_CODECS = ["auto", "h264", "av1"]
CONTAINER_EXTENSIONS = {"mp4": "mp4", "mkv": "mkv", "webm": "webm"}


def _metadata_disabled() -> bool:
    """Honor ComfyUI's --disable-metadata flag when the CLI args are available."""
    try:
        from comfy.cli_args import args  # pyright: ignore[reportMissingImports]
    except Exception:
        return False
    return bool(getattr(args, "disable_metadata", False))


def resolve_video_format(format_name: str, codec_name: str) -> str:
    """Resolve the 'auto' container the same way ComfyUI's Save Video node does."""
    if format_name == "auto":
        return "webm" if codec_name == "av1" else "mp4"
    return format_name


def to_video_container(format_name: str):
    if _ComfyTypes is not None:
        return _ComfyTypes.VideoContainer(format_name)
    return format_name


def to_video_codec(codec_name: str):
    if _ComfyTypes is not None:
        return _ComfyTypes.VideoCodec(codec_name)
    return codec_name


class SaveVideoLM(SaveImageLM):
    """Save VIDEO inputs with compact, A1111-style generation metadata.

    ComfyUI's built-in Save Video node dumps the full workflow and prompt JSON
    into the container metadata, which can make the write fail for large
    workflows. This node only writes the LoRA Manager ``parameters`` string
    (and optionally the workflow), so the saved file stays clean.
    """

    NAME = "Save Video (LoraManager)"
    CATEGORY = "Lora Manager/utils"
    DESCRIPTION = (
        "Save videos with compact generation metadata instead of the full "
        "workflow/prompt dump written by the built-in Save Video node."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "The video to save."}),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "video/ComfyUI",
                        "tooltip": "Base filename for saved videos. Supports format patterns like %seed%, %width%, %height%, %model%, etc.",
                    },
                ),
                "format": (
                    VIDEO_FORMATS,
                    {
                        "tooltip": "Output container. Auto uses MP4 for Auto/H.264 and WebM for AV1.",
                    },
                ),
                "codec": (
                    VIDEO_CODECS,
                    {
                        "tooltip": "Video codec. Auto picks H.264 for MP4/MKV and AV1 for WebM.",
                    },
                ),
            },
            "optional": {
                "embed_workflow": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled, stores the complete workflow in the video metadata so it can be dragged back into ComfyUI. Disabled by default to keep the container metadata small.",
                    },
                ),
                "save_with_metadata": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, writes the generation parameters (prompt, seed, model, LoRAs...) as a compact 'parameters' tag. Disable to skip writing generation metadata.",
                    },
                ),
                "add_loras_to_prompt": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled, appends the LoRA syntax line (e.g. <lora:name:strength>) after the positive prompt in the saved metadata.",
                    },
                ),
                "add_counter_to_filename": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Adds an incremental counter to filenames to prevent overwriting previous videos.",
                    },
                ),
            },
            "hidden": {
                "id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "process_video"
    OUTPUT_NODE = True

    def build_video_metadata(
        self,
        metadata: str,
        extra_pnginfo: Optional[dict],
        embed_workflow: bool,
        save_with_metadata: bool,
    ) -> Optional[Dict[str, Any]]:
        """Build the container metadata tags. Returns None when nothing should be written."""
        if _metadata_disabled():
            return None

        tags: Dict[str, Any] = {}
        if save_with_metadata and metadata:
            tags["parameters"] = metadata
        if embed_workflow and extra_pnginfo is not None:
            workflow = extra_pnginfo.get("workflow")
            if workflow is not None:
                tags["workflow"] = workflow
        return tags or None

    def save_video(
        self,
        video,
        filename_prefix,
        format,
        codec,
        id,
        prompt=None,
        extra_pnginfo=None,
        embed_workflow=False,
        save_with_metadata=True,
        add_loras_to_prompt=False,
        add_counter_to_filename=True,
    ):
        """Encode the video to the output directory and return the UI result entry."""
        raw_metadata = get_metadata()
        metadata_dict = MetadataProcessor.to_dict(raw_metadata, id)
        metadata = self.format_metadata(metadata_dict, add_loras_to_prompt)

        filename_prefix = self.format_filename(filename_prefix, metadata_dict)

        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, _processed_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, width, height
            )
        )
        os.makedirs(full_output_folder, exist_ok=True)

        format_name = resolve_video_format(format, codec)
        extension = CONTAINER_EXTENSIONS[format_name]

        base_filename = filename
        if add_counter_to_filename:
            base_filename += f"_{counter:05}_"
        file = f"{base_filename}.{extension}"
        file_path = os.path.join(full_output_folder, file)

        saved_metadata = self.build_video_metadata(
            metadata, extra_pnginfo, embed_workflow, save_with_metadata
        )

        video.save_to(
            file_path,
            format=to_video_container(format_name),
            codec=to_video_codec(codec),
            metadata=saved_metadata,
        )

        return {"filename": file, "subfolder": subfolder, "type": self.type}

    def process_video(
        self,
        video,
        id,
        filename_prefix="video/ComfyUI",
        format="auto",
        codec="auto",
        prompt=None,
        extra_pnginfo=None,
        embed_workflow=False,
        save_with_metadata=True,
        add_loras_to_prompt=False,
        add_counter_to_filename=True,
    ):
        """Save the video and pass it through unchanged."""
        os.makedirs(self.output_dir, exist_ok=True)

        result = self.save_video(
            video,
            filename_prefix,
            format,
            codec,
            id,
            prompt,
            extra_pnginfo,
            embed_workflow,
            save_with_metadata,
            add_loras_to_prompt,
            add_counter_to_filename,
        )

        return {
            "result": (video,),
            "ui": {"images": [result], "animated": (True,)},
        }
