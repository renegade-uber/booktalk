import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from protocols import (
    Passage, 
    BookLoaderProtocol, 
    LLMServiceProtocol, 
    VectorStoreProtocol, 
    AnsweringServiceProtocol,
)
from openai import OpenAI
from typing import Dict, Any, List

"""
This file follows the Presentation-Domain-Data architecture:
- Presentation: To run the application via terminal or the chainlit app
- Domain: The business logic services
- Data: The external data dependencies
"""


"""
Data Layer classes
"""
class BookLoader(BookLoaderProtocol):
    def load(self, file_path: str) -> str:
        """Load book content from a file path"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.pdf':
            return self._extract_text_from_pdf(file_path)
        else:
            raise ValueError("Unsupported file type: must be .txt or .pdf")
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

class OpenAIService(LLMServiceProtocol):
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.0):
        self.model_name = model_name
        self.temperature = temperature
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=api_key)
    
    def generate_answer(self, question: str, context: List[str]) -> str:
        prompt = f"""You are a helpful assistant. ONLY use the information provided in the context below to answer the user's question. 
Do not add any outside knowledge or make assumptions. If the answer is not in the context, respond with "I don't know based on the provided information."
When answering questions, generate thorough, multi-paragraph responses that cite and summarize all relevant context chunks. Do not be brief.
Start with a short summary answering the question, then provide additional details and context from the provided information.
Context:
{chr(10).join(context)}

Question:
{question}

Answer:"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()


class FAISSIndexer(VectorStoreProtocol):
    def __init__(self, model):
        self.model = model
        # Determine dimension by encoding a small dummy text
        dummy_embedding = model.encode(["Determine dimension"])
        self.dim = dummy_embedding[0].shape[0]
        # Initialize empty index with correct dimensions
        self.index = faiss.IndexFlatL2(self.dim)
        self.passages = []
    
    def index_passages(self, passages: list[Passage]) -> Any:
        """Create or update vector index with passages"""
        if not passages:
            return self.index
            
        # Encode new passages
        embeddings = self.model.encode([p.text for p in passages], show_progress_bar=True)
        
        # Add to index
        self.index.add(np.array(embeddings))
        
        # Append to existing passages
        self.passages.extend(passages)
            
        return len(self.passages)
    
    def search(self, query: str, top_k: int = 10) -> list[Passage]:
        """Search for relevant passages"""
        query_vec = self.model.encode([query])
        D, I = self.index.search(np.array(query_vec), top_k)
        return [self.passages[i] for i in I[0]]

"""
Domain Layer classes
"""
class BookTalkService(AnsweringServiceProtocol):
    def __init__(
            self, 
            book_loader: BookLoaderProtocol, 
            vector_store: VectorStoreProtocol, 
            llm_service: LLMServiceProtocol, 
            chunk_size: int=300
    ):
        self.book_loader = book_loader
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.chunk_size = chunk_size
        self.indexed_books = set()
    
    @staticmethod
    def create_instance(chunk_size=300):
        book_loader = BookLoader()
        vector_store = FAISSIndexer(SentenceTransformer('all-MiniLM-L6-v2'))
        llm_service = OpenAIService()

        return BookTalkService(book_loader, vector_store, llm_service, chunk_size)

    def ingest_book(self, file_path: str) -> bool:
        """Process and index a book for later querying"""
        try:
            # Load the book content
            book_text = self.book_loader.load(file_path)
            
            # Split into chunks and create Passage objects
            passages = []
            chunks = self.chunk_text(book_text, self.chunk_size)
            for chunk in chunks:
                passages.append(Passage(text=chunk, source=file_path))
            
            # Index the passages
            self.vector_store.index_passages(passages)
            
            # Track that we've indexed this book
            self.indexed_books.add(file_path)
            return True
        except Exception as e:
            print(f"Error ingesting book: {e}")
            return False
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question about ingested books, returning answer and sources"""
        # Search for relevant chunks
        relevant_passages = self.vector_store.search(question)
        
        # Extract text from passages for the LLM
        relevant_texts = [passage.text for passage in relevant_passages]
        
        # Generate answer using LLM
        answer = self.llm_service.generate_answer(question, relevant_texts)
        
        # Return answer and sources
        return {
            "answer": answer,
            "sources": relevant_passages
        }
    
    @staticmethod
    # Step 2: Split text into chunks (e.g. ~300 words)
    def chunk_text(text, max_words=300):
        paragraphs = text.split("\n")
        chunks, current_chunk = [], []
        word_count = 0
        for para in paragraphs:
            words = para.split()
            if not words:
                continue
            if word_count + len(words) > max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk, word_count = [], 0
            current_chunk.extend(words)
            word_count += len(words)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks



"""
Presentation Layer
"""

# Chainlit application
import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    books_folder = os.path.expanduser("~/Documents/books")
    book_files = [f for f in os.listdir(books_folder) if f.endswith(('.pdf', '.txt'))][:9]
    
    if not book_files:
        await cl.Message(content="No PDF or TXT files found in your books folder.").send()
        return
    
    # Initialize services
    booktalk_service = BookTalkService.create_instance()
    
    # Ingest all books
    for book_file in book_files:
        book_path = os.path.join(books_folder, book_file)
        try:
            booktalk_service.ingest_book(book_path)
            await cl.Message(content=f"Added {book_file}").send()
        except Exception as e:
            await cl.Message(content=f"Error loading {book_file}: {str(e)}").send()
            continue

    await cl.Message(content="Books indexed successfully. Ask your questions!").send()
    cl.user_session.set("booktalk_service", booktalk_service)

@cl.on_message
async def on_message(message: cl.Message):
    booktalk_service = cl.user_session.get("booktalk_service")
    if not booktalk_service:
        await cl.Message(content="Please wait for book indexing to complete.").send()
        return
    answer = booktalk_service.answer_question(message.content)
    await cl.Message(content=answer['answer']).send()
    sources_content = "\n\n".join([f"Source {i+1}:\n{source.text}" for i, source in enumerate(answer['sources'])])
    sources = cl.Text(content=sources_content, display="inline")
    await cl.Message(content="Sources:", elements=[sources]).send()


# Terminal application
def main():
    books_folder = os.path.expanduser("~/Documents/books")
    book_files = [f for f in os.listdir(books_folder) if f.endswith(('.pdf', '.txt'))][:9]
    
    if not book_files:
        print("No PDF or TXT files found in your books folder.")
        return
    
    print("Available books:")
    for i, book in enumerate(book_files, 1):
        print(f"{i}. {book}")
    
    selection = input("Enter book numbers to index (1,2,3) or leave empty for all: ").strip()
    
    if selection:
        try:
            selected_indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
            selected_books = [book_files[idx] for idx in selected_indices if 0 <= idx < len(book_files)]
        except (ValueError, IndexError):
            print("Invalid selection. Using all available books.")
            selected_books = book_files
    else:
        selected_books = book_files
    
    # Initialize services
    booktalk_service = BookTalkService.create_instance()
    
    # Ingest selected books
    for book_file in selected_books:
        book_path = os.path.join(books_folder, book_file)
        print(f"Loading {book_file}...")
        try:
            booktalk_service.ingest_book(book_path)
            print(f"Added {book_file}")
        except Exception as e:
            print(f"Error loading {book_file}: {e}")
    
    print("Books indexed successfully. Ask your question below (type 'exit' to quit).\n")
    
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ('exit', 'quit'): break
        answer = booktalk_service.answer_question(question)
        print("\n--- Sources ---")
        for source in answer['sources']:
            print(f"\n{source.text}\n")

        print("\n--- Answer ---")
        print(f"\n{answer['answer']}\n")

if __name__ == '__main__':
    main()
