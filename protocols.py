"""
Domain Protocols
"""
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable

@runtime_checkable
class AnsweringServiceProtocol(Protocol):
    def ingest_book(self, file_path: str) -> bool:
        """Process and index a book for later querying"""
        ...
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question about ingested books, returning answer and sources"""
        ...

"""
Data Protocols and Entities
"""
@runtime_checkable
class BookLoaderProtocol(Protocol):
    def load(self, file_path: str) -> str:
        """Load book content from a file path"""
        ...

class Passage:
    """Represents a passage of text from a document"""
    def __init__(self, text: str, source: str, metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.source = source
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        return f"Passage from {self.source}: {self.text[:50]}..."

@runtime_checkable
class VectorStoreProtocol(Protocol):
    def index_passages(self, passages: List[Passage]) -> Any:
        """Create vector index from passages"""
        ...
    
    def search(self, query: str, top_k: int = 10) -> List[Passage]:
        """Search for relevant passages"""
        ...

@runtime_checkable
class LLMServiceProtocol(Protocol):
    def generate_answer(self, question: str, context: List[str]) -> str:
        """Generate an answer using an LLM based on question and context"""
        ...
