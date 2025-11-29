"""
Deprecated multi-node runner. Run individual smoke files instead:
- test_lora_lo_r_a_selector.py
- test_lora_lo_r_a_selector_x_l.py
"""

if __name__ == "__main__":
    import sys
    sys.stderr.write("Use the per-node scripts: test_lora_lo_r_a_selector.py, test_lora_lo_r_a_selector_x_l.py.\n")
    raise SystemExit(0)
