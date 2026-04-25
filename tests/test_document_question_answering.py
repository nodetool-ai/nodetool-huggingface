import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.nodes.huggingface.document_question_answering import DocumentQuestionAnswering


def test_document_qa_default_model():
    node = DocumentQuestionAnswering()
    assert node.model.repo_id == "impira/layoutlm-document-qa"


def test_document_qa_recommended_models():
    models = DocumentQuestionAnswering.get_recommended_models()
    assert len(models) > 0
    repo_ids = [m.repo_id for m in models]
    assert any("layoutlm" in r for r in repo_ids)
    assert any("donut" in r for r in repo_ids)


def test_document_qa_required_inputs():
    assert DocumentQuestionAnswering.required_inputs(DocumentQuestionAnswering()) == ["image", "question"]


def test_document_qa_title():
    assert DocumentQuestionAnswering.get_title() == "Document Question Answering"
