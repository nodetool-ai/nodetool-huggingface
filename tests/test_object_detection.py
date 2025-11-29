"""
Deprecated multi-node runner. Run individual smoke files instead:
- test_object_detection_object_detection.py
- test_object_detection_visualize_object_detection.py
- test_object_detection_zero_shot_object_detection.py
"""

if __name__ == "__main__":
    import sys
    sys.stderr.write("Use the per-node scripts: test_object_detection_object_detection.py, test_object_detection_visualize_object_detection.py, test_object_detection_zero_shot_object_detection.py.\n")
    raise SystemExit(0)
