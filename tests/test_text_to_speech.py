"""
Deprecated multi-node runner. Run individual smoke files instead:
- test_text_to_speech_bark.py
- test_text_to_speech_kokoro_t_t_s.py
- test_text_to_speech_text_to_speech.py
"""

if __name__ == "__main__":
    import sys
    sys.stderr.write("Use the per-node scripts: test_text_to_speech_bark.py, test_text_to_speech_kokoro_t_t_s.py, test_text_to_speech_text_to_speech.py.\n")
    raise SystemExit(0)
