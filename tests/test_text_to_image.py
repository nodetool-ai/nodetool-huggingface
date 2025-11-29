"""
Deprecated multi-node runner. Run individual smoke files instead:
- test_text_to_image_chroma.py
- test_text_to_image_flux.py
- test_text_to_image_flux_control.py
- test_text_to_image_qwen_image.py
- test_text_to_image_sdxl.py
"""

if __name__ == "__main__":
    import sys
    sys.stderr.write("Use the per-node scripts listed above.\n")
    raise SystemExit(0)
