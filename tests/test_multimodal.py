"""
Deprecated multi-node runner. Run individual smoke files instead:
- test_multimodal_image_to_text.py
- test_multimodal_visual_question_answering.py
"""

if __name__ == "__main__":
    import sys
    sys.stderr.write("Use the per-node scripts: test_multimodal_image_to_text.py, test_multimodal_visual_question_answering.py.\n")
    raise SystemExit(0)
