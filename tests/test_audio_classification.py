"""
Deprecated multi-node runner. Run individual smoke files instead:
- test_audio_classification_audio_classifier.py
- test_audio_classification_zero_shot_audio_classifier.py
"""

if __name__ == "__main__":
    import sys
    sys.stderr.write("Use the per-node scripts: test_audio_classification_audio_classifier.py, test_audio_classification_zero_shot_audio_classifier.py.\n")
    raise SystemExit(0)
