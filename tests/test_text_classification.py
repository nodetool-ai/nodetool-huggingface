"""
Deprecated multi-node runner. Run individual smoke files instead:
- test_text_classification_text_classifier.py
- test_text_classification_zero_shot_text_classifier.py
"""

if __name__ == "__main__":
    import sys
    sys.stderr.write("Use the per-node scripts: test_text_classification_text_classifier.py, test_text_classification_zero_shot_text_classifier.py.\n")
    raise SystemExit(0)
