from nodetool.huggingface.tts_adapter_manifest import get_tts_adapter_info


def test_bark_artifact_names_the_files_the_adapter_loads():
    info = get_tts_adapter_info("suno/bark")

    assert info.artifact_ref is not None
    assert info.artifact_ref.allow_patterns == ["*.bin", "*.json", "*.txt"]


def test_unlisted_repository_downloads_without_patterns():
    info = get_tts_adapter_info("facebook/mms-tts-fra")

    assert info.artifact_ref is not None
    assert info.artifact_ref.allow_patterns is None
