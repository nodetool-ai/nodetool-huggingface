from nodetool.nodes.nodetool.document import DocumentRef
from nodetool.workflows.processing_context import ProcessingContext
from typing import TypedDict, AsyncGenerator
from nodetool.workflows.base_node import BaseNode
from pydantic import Field

class SplitSentences(BaseNode):
    """
    Splits text into sentences using LangChain's SentenceTransformersTokenTextSplitter.
    sentences, split, nlp

    Use cases:
    - Natural sentence-based text splitting
    - Creating semantically meaningful chunks
    - Processing text for sentence-level analysis
    """

    document: DocumentRef = Field(default=DocumentRef())
    chunk_size: int = Field(
        default=40,
        description="Maximum number of tokens per chunk",
    )
    chunk_overlap: int = Field(
        default=5,
        description="Number of tokens to overlap between chunks",
    )

    @classmethod
    def get_title(cls):
        return "Split into Sentences"

    class OutputType(TypedDict):
        text: str
        source_id: str
        start_index: int

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        from langchain_text_splitters import SentenceTransformersTokenTextSplitter
        from langchain_core.documents import Document

        splitter = SentenceTransformersTokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )

        docs = splitter.split_documents([Document(page_content=self.document.data)])

        for i, doc in enumerate(docs):
            yield {
                "text": doc.page_content,
                "source_id": f"{self.document.uri}:{i}",
                "start_index": doc.metadata.get("start_index", 0),
            }