"""
Deprecated multi-node runner. Run individual smoke files instead:
- test_text_to_video_cog_video_x.py
- test_text_to_video_wan__t2_v.py
"""

if __name__ == "__main__":
    import sys
    sys.stderr.write("Use the per-node scripts: test_text_to_video_cog_video_x.py, test_text_to_video_wan__t2_v.py.\n")
    raise SystemExit(0)
