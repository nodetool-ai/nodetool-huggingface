"""
Deprecated multi-node runner. Run individual smoke files instead:
- test_question_answering_question_answering.py
- test_question_answering_table_question_answering.py
"""

if __name__ == "__main__":
    import sys
    sys.stderr.write("Use the per-node scripts: test_question_answering_question_answering.py, test_question_answering_table_question_answering.py.\n")
    raise SystemExit(0)
