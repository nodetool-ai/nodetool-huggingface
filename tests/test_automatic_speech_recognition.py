"""
Deprecated multi-node runner. Run individual smoke files instead:
- test_automatic_speech_recognition_chunks_to_s_r_t.py
- test_automatic_speech_recognition_whisper.py
"""

if __name__ == "__main__":
    import sys
    sys.stderr.write("Use the per-node scripts: test_automatic_speech_recognition_chunks_to_s_r_t.py, test_automatic_speech_recognition_whisper.py.\n")
    raise SystemExit(0)
