"""
Deprecated multi-node runner. Run individual smoke files instead:
- test_image_segmentation_find_segment.py
- test_image_segmentation_s_a_m2_segmentation.py
- test_image_segmentation_segmentation.py
- test_image_segmentation_visualize_segmentation.py
"""

if __name__ == "__main__":
    import sys
    sys.stderr.write("Use the per-node scripts: test_image_segmentation_find_segment.py, test_image_segmentation_s_a_m2_segmentation.py, test_image_segmentation_segmentation.py, test_image_segmentation_visualize_segmentation.py.\n")
    raise SystemExit(0)
