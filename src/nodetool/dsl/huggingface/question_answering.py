from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class QuestionAnswering(GraphNode):
    """
    Answers questions based on a given context.
    text, question answering, natural language processing

    Use cases:
    - Automated customer support
    - Information retrieval from documents
    - Reading comprehension tasks
    - Enhancing search functionality
    """

    model: types.HFQuestionAnswering | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFQuestionAnswering(
            type="hf.question_answering",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model ID to use for question answering",
    )
    context: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The context or passage to answer questions from"
    )
    question: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The question to be answered based on the context"
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.question_answering.QuestionAnswering"


class TableQuestionAnswering(GraphNode):
    """
    Answers questions based on tabular data.
    table, question answering, natural language processing

    Use cases:
    - Querying databases using natural language
    - Analyzing spreadsheet data with questions
    - Extracting insights from tabular reports
    - Automated data exploration
    """

    model: types.HFTableQuestionAnswering | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFTableQuestionAnswering(
            type="hf.table_question_answering",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model ID to use for table question answering",
    )
    dataframe: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="The input table to query",
    )
    question: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The question to be answered based on the table"
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.question_answering.TableQuestionAnswering"
