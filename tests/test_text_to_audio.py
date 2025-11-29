"""
Deprecated multi-node runner. Run individual smoke files instead:
- test_text_to_audio_audio_l_d_m.py
- test_text_to_audio_audio_l_d_m2.py
- test_text_to_audio_dance_diffusion.py
- test_text_to_audio_music_gen.py
- test_text_to_audio_music_l_d_m.py
- test_text_to_audio_stable_audio_node.py
"""

if __name__ == "__main__":
    import sys
    sys.stderr.write("Use the per-node scripts: test_text_to_audio_audio_l_d_m.py, test_text_to_audio_audio_l_d_m2.py, test_text_to_audio_dance_diffusion.py, test_text_to_audio_music_gen.py, test_text_to_audio_music_l_d_m.py, test_text_to_audio_stable_audio_node.py.\n")
    raise SystemExit(0)
