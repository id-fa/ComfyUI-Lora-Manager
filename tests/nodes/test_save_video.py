import os

import pytest

from py.nodes.save_video import SaveVideoLM


class _FakeVideo:
    """Minimal stand-in for ComfyUI's VideoInput."""

    def __init__(self, width=640, height=360):
        self._width = width
        self._height = height
        self.calls = []

    def get_dimensions(self):
        return self._width, self._height

    def save_to(self, path, format=None, codec=None, metadata=None, **kwargs):
        self.calls.append(
            {"path": path, "format": format, "codec": codec, "metadata": metadata}
        )
        with open(path, "wb") as handle:
            handle.write(b"video")


def _configure_save_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "folder_paths.get_output_directory", lambda: str(tmp_path), raising=False
    )
    monkeypatch.setattr(
        "folder_paths.get_save_image_path",
        lambda *_args, **_kwargs: (str(tmp_path), "sample", 1, "", "sample"),
        raising=False,
    )


def _configure_metadata(monkeypatch, metadata_dict):
    monkeypatch.setattr("py.nodes.save_video.get_metadata", lambda: {"raw": "metadata"})
    monkeypatch.setattr(
        "py.nodes.save_video.MetadataProcessor.to_dict",
        lambda raw_metadata, node_id: metadata_dict,
    )
    monkeypatch.setattr("py.nodes.save_video._metadata_disabled", lambda: False)


def _format_value(value):
    return getattr(value, "value", value)


def test_save_video_writes_only_compact_parameters_by_default(monkeypatch, tmp_path):
    _configure_save_paths(monkeypatch, tmp_path)
    _configure_metadata(monkeypatch, {"prompt": "prompt text", "seed": 123})

    video = _FakeVideo()
    node = SaveVideoLM()
    result = node.process_video(
        video,
        id="node-1",
        prompt={"1": {"class_type": "KSampler"}},
        extra_pnginfo={"workflow": {"nodes": []}, "other": "junk"},
    )

    assert len(video.calls) == 1
    call = video.calls[0]
    assert call["path"] == os.path.join(str(tmp_path), "sample_00001_.mp4")
    assert call["metadata"] == {"parameters": "prompt text\nSeed: 123, Version: ComfyUI"}
    assert _format_value(call["format"]) == "mp4"
    assert _format_value(call["codec"]) == "auto"

    assert result["result"] == (video,)
    assert result["ui"] == {
        "images": [{"filename": "sample_00001_.mp4", "subfolder": "", "type": "output"}],
        "animated": (True,),
    }


def test_save_video_embeds_only_workflow_when_enabled(monkeypatch, tmp_path):
    _configure_save_paths(monkeypatch, tmp_path)
    _configure_metadata(monkeypatch, {"prompt": "prompt text", "seed": 123})

    video = _FakeVideo()
    node = SaveVideoLM()
    workflow = {"nodes": [{"id": 1}]}
    node.process_video(
        video,
        id="node-1",
        prompt={"1": {"class_type": "KSampler"}},
        extra_pnginfo={"workflow": workflow, "other": "junk"},
        embed_workflow=True,
        save_with_metadata=False,
    )

    assert video.calls[0]["metadata"] == {"workflow": workflow}


def test_save_video_passes_no_metadata_when_everything_disabled(monkeypatch, tmp_path):
    _configure_save_paths(monkeypatch, tmp_path)
    _configure_metadata(monkeypatch, {"prompt": "prompt text", "seed": 123})

    video = _FakeVideo()
    node = SaveVideoLM()
    node.process_video(
        video,
        id="node-1",
        extra_pnginfo={"workflow": {"nodes": []}},
        save_with_metadata=False,
    )

    assert video.calls[0]["metadata"] is None


def test_save_video_respects_disable_metadata_flag(monkeypatch, tmp_path):
    _configure_save_paths(monkeypatch, tmp_path)
    _configure_metadata(monkeypatch, {"prompt": "prompt text", "seed": 123})
    monkeypatch.setattr("py.nodes.save_video._metadata_disabled", lambda: True)

    video = _FakeVideo()
    node = SaveVideoLM()
    node.process_video(
        video,
        id="node-1",
        extra_pnginfo={"workflow": {"nodes": []}},
        embed_workflow=True,
    )

    assert video.calls[0]["metadata"] is None


def test_save_video_appends_loras_to_prompt_when_enabled(monkeypatch, tmp_path):
    _configure_save_paths(monkeypatch, tmp_path)
    _configure_metadata(
        monkeypatch,
        {
            "prompt": "prompt text",
            "seed": 123,
            "loras": "<lora:styleA:0.8>",
        },
    )

    video = _FakeVideo()
    node = SaveVideoLM()
    node.process_video(video, id="node-1", add_loras_to_prompt=True)

    parameters = video.calls[0]["metadata"]["parameters"]
    assert parameters.startswith("prompt text\n<lora:styleA:0.8>\n")


@pytest.mark.parametrize(
    ("format_name", "codec_name", "expected_format", "expected_ext"),
    [
        ("auto", "auto", "mp4", "mp4"),
        ("auto", "h264", "mp4", "mp4"),
        ("auto", "av1", "webm", "webm"),
        ("mkv", "auto", "mkv", "mkv"),
        ("webm", "av1", "webm", "webm"),
    ],
)
def test_save_video_resolves_container_and_extension(
    monkeypatch, tmp_path, format_name, codec_name, expected_format, expected_ext
):
    _configure_save_paths(monkeypatch, tmp_path)
    _configure_metadata(monkeypatch, {"prompt": "p", "seed": 1})

    video = _FakeVideo()
    node = SaveVideoLM()
    result = node.process_video(video, id="node-1", format=format_name, codec=codec_name)

    call = video.calls[0]
    assert _format_value(call["format"]) == expected_format
    assert _format_value(call["codec"]) == codec_name
    assert call["path"].endswith(f"sample_00001_.{expected_ext}")
    assert result["ui"]["images"][0]["filename"] == f"sample_00001_.{expected_ext}"


def test_save_video_can_skip_counter(monkeypatch, tmp_path):
    _configure_save_paths(monkeypatch, tmp_path)
    _configure_metadata(monkeypatch, {"prompt": "p", "seed": 1})

    video = _FakeVideo()
    node = SaveVideoLM()
    node.process_video(video, id="node-1", add_counter_to_filename=False)

    assert video.calls[0]["path"] == os.path.join(str(tmp_path), "sample.mp4")


def test_save_video_applies_filename_patterns(monkeypatch, tmp_path):
    _configure_metadata(monkeypatch, {"prompt": "p", "seed": 4242})
    monkeypatch.setattr(
        "folder_paths.get_output_directory", lambda: str(tmp_path), raising=False
    )
    seen = {}

    def fake_get_save_image_path(prefix, output_dir, width, height):
        seen["prefix"] = prefix
        seen["size"] = (width, height)
        return (str(tmp_path), prefix, 1, "", prefix)

    monkeypatch.setattr(
        "folder_paths.get_save_image_path", fake_get_save_image_path, raising=False
    )

    video = _FakeVideo(width=1280, height=720)
    node = SaveVideoLM()
    node.process_video(video, id="node-1", filename_prefix="clip_%seed%")

    assert seen["prefix"] == "clip_4242"
    assert seen["size"] == (1280, 720)


def test_save_video_input_types_and_node_metadata():
    inputs = SaveVideoLM.INPUT_TYPES()

    assert inputs["required"]["video"][0] == "VIDEO"
    assert inputs["required"]["format"][0] == ["auto", "mp4", "mkv", "webm"]
    assert inputs["required"]["codec"][0] == ["auto", "h264", "av1"]
    assert inputs["optional"]["embed_workflow"][1]["default"] is False
    assert inputs["optional"]["save_with_metadata"][1]["default"] is True
    assert inputs["hidden"] == {
        "id": "UNIQUE_ID",
        "prompt": "PROMPT",
        "extra_pnginfo": "EXTRA_PNGINFO",
    }
    assert SaveVideoLM.NAME == "Save Video (LoraManager)"
    assert SaveVideoLM.RETURN_TYPES == ("VIDEO",)
    assert SaveVideoLM.FUNCTION == "process_video"
    assert SaveVideoLM.OUTPUT_NODE is True
