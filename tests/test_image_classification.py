"""
Deprecated multi-node runner. Run individual smoke files instead:
- test_image_classification_image_classifier.py
- test_image_classification_zero_shot_image_classifier.py
"""

if __name__ == "__main__":
    import sys
    sys.stderr.write("Use the per-node scripts: test_image_classification_image_classifier.py, test_image_classification_zero_shot_image_classifier.py.\n")
    raise SystemExit(0)
