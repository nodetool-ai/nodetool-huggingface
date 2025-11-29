"""
Deprecated multi-node runner. Run individual smoke files instead:
- test_image_to_text_image_to_text.py
- test_image_to_text_load_image_to_text_model.py
"""

if __name__ == "__main__":
    import sys
    sys.stderr.write("Use the per-node scripts: test_image_to_text_image_to_text.py, test_image_to_text_load_image_to_text_model.py.\n")
    raise SystemExit(0)
